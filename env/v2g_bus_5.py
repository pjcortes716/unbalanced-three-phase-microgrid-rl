from dataclasses import dataclass
import operator

import numpy as np


@dataclass(frozen=True)
class V2GActionResolution:
    """Resultado inmutable de resolver una acción V2G sin mutar el camión."""

    t: int
    connected: bool
    aP_requested: float
    aQ_requested: float
    aP_applied: float
    aQ_applied: float
    P_grid_kw: float
    Q_grid_kvar: float
    q_max_kvar: float
    E_before_kWh: float
    E_next_kWh: float
    soc_before: float
    soc_next: float


class V2GTruck:
    """Camión V2G con disponibilidad semanal y consulta temporal explícita.

    El ambiente futuro conserva la autoridad temporal: consulta
    ``state_at(env.t)``, aplica ``apply_action_at(env.t, ...)`` y cierra el
    intervalo con ``advance_interval(env.t)``. ``self.t`` se conserva como
    comprobación de sincronía y para compatibilidad con ``step``.

    La conexión física se decidirá en el ambiente como un equipo trifásico
    balanceado; este componente solo modela potencia total P/Q y SOC.
    """

    horizon_hours = 168

    def __init__(
        self,
        availability_7x24: np.ndarray,
        seed: int = 0,
        s_max_kva: float = 150.0,
        p_ch_max_kw: float = 150.0,
        p_dch_max_kw: float = 150.0,
        eta_ch: float = 0.95,
        eta_dch: float = 0.95,
        capacity_kwh: float = 315.0,
        soc_min: float = 0.10,
        soc_target_dep: float = 0.80,
        soc_arr_min: float = 0.20,
        soc_arr_max: float = 0.40,
        allow_q_when_disconnected: bool = False,
    ):
        self.rng = np.random.default_rng(seed)

        availability_7x24 = np.asarray(availability_7x24, dtype=int)
        if availability_7x24.shape != (7, 24):
            raise ValueError("availability_7x24 debe tener shape (7, 24).")
        if not np.all((availability_7x24 == 0) | (availability_7x24 == 1)):
            raise ValueError("availability_7x24 debe contener solamente 0 y 1.")
        self.avail_168 = availability_7x24.reshape(-1).astype(int)
        self.avail_7x24 = availability_7x24.copy()

        self.s_max = float(s_max_kva)
        self.p_ch_max = float(p_ch_max_kw)
        self.p_dch_max = float(p_dch_max_kw)
        self.eta_ch = float(eta_ch)
        self.eta_dch = float(eta_dch)
        self.C = float(capacity_kwh)
        self.soc_min = float(soc_min)
        self.soc_target_dep = float(soc_target_dep)
        self.soc_arr_min = float(soc_arr_min)
        self.soc_arr_max = float(soc_arr_max)
        if self.soc_target_dep <= self.soc_min:
            raise ValueError("soc_target_dep debe ser mayor que soc_min.")
        self.allow_q_when_disconnected = bool(allow_q_when_disconnected)

        self.t = 0
        self.connected = False
        self.soc = 0.0
        self.P_grid_kw = 0.0
        self.Q_grid_kvar = 0.0
        self.last_qmax = self.s_max
        self._action_applied_t: int | None = None

    def _configure_rng(self, *, rng=None, seed: int | None = None) -> None:
        if rng is not None and seed is not None:
            raise ValueError("Pasa rng o seed, no ambos.")
        if rng is not None:
            if not hasattr(rng, "uniform"):
                raise TypeError("rng debe ofrecer uniform() como un Generator de NumPy.")
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)

    def _validate_time(self, t: int) -> int:
        try:
            t = operator.index(t)
        except TypeError as exc:
            raise TypeError("t debe ser un índice entero de hora.") from exc
        if not 0 <= t < self.horizon_hours:
            raise IndexError(f"t={t} está fuera del horizonte [0, {self.horizon_hours - 1}].")
        return int(t)

    def available_at(self, t: int) -> bool:
        """Disponibilidad programada de ``t``; no cambia ningún estado."""
        return bool(self.avail_168[self._validate_time(t)])

    def _hours_to_transition(self, t: int, target_connected: bool) -> int:
        for dt in range(1, self.horizon_hours - t):
            if bool(self.avail_168[t + dt]) is target_connected:
                return dt
        return 0

    def _assert_current_time(self, t: int) -> int:
        t = self._validate_time(t)
        if t != self.t:
            raise ValueError(
                f"El estado del truck está en t={self.t}; no se puede usar t={t}. "
                "Avanza ambos relojes exactamente una vez por intervalo."
            )
        if self.connected != self.available_at(t):
            raise RuntimeError("Conexión inconsistente con la disponibilidad de la hora actual.")
        return t

    def _sample_arrival_soc(self) -> None:
        self.soc = float(self.rng.uniform(self.soc_arr_min, self.soc_arr_max))
        self.soc = float(np.clip(self.soc, self.soc_min, 1.0))

    def reset(self, *, rng=None, seed: int | None = None) -> np.ndarray:
        """Reinicia el truck y opcionalmente sustituye el RNG."""
        self._configure_rng(rng=rng, seed=seed)
        self.t = 0
        self.connected = self.available_at(0)
        self._sample_arrival_soc()
        self.P_grid_kw = 0.0
        self.Q_grid_kvar = 0.0
        self.last_qmax = self.s_max if self.connected else (
            self.s_max if self.allow_q_when_disconnected else 0.0
        )
        self._action_applied_t = None
        return self.state_at(0)

    def state_at(self, t: int) -> np.ndarray:
        """Estado no mutante de la hora ``t`` sincronizada con el ambiente.

        El vector oficial tiene cinco componentes: ``connected``, ``soc``,
        horas a salida / 168, horas a llegada / 168 y brecha SOC normalizada.
        """
        t = self._assert_current_time(t)
        connected = 1.0 if self.available_at(t) else 0.0
        if connected:
            h_to_dep = self._hours_to_transition(t, False)
            h_to_arr = 0
        else:
            h_to_dep = 0
            h_to_arr = self._hours_to_transition(t, True)

        gap = max(0.0, self.soc_target_dep - self.soc)
        gap_norm = gap / (self.soc_target_dep - self.soc_min)
        return np.array(
            [connected, self.soc, h_to_dep / 168.0, h_to_arr / 168.0, gap_norm],
            dtype=float,
        )

    def _terminal_state(self) -> np.ndarray:
        gap = max(0.0, self.soc_target_dep - self.soc)
        gap_norm = gap / (self.soc_target_dep - self.soc_min)
        return np.array([0.0, self.soc, 0.0, 0.0, gap_norm], dtype=float)

    def get_state(self, t: int | None = None) -> np.ndarray:
        """Consulta no mutante; sin ``t`` conserva el uso legacy."""
        if t is None:
            if self.t >= self.horizon_hours:
                return self._terminal_state()
            t = self.t
        return self.state_at(t)

    def _clip_p_by_soc(self, p_cmd: float) -> float:
        p_cmd = float(np.clip(p_cmd, -self.p_dch_max, self.p_ch_max))
        if p_cmd >= 0.0:
            energy_room = (1.0 - self.soc) * self.C
            p_cmd = min(p_cmd, energy_room / max(self.eta_ch, 1e-9))
        else:
            energy_available = (self.soc - self.soc_min) * self.C
            p_cmd = max(p_cmd, -energy_available * max(self.eta_dch, 1e-9))
        return float(p_cmd)

    def _resolve_soc_after_grid_power(self, p_grid_kw: float) -> tuple[float, float]:
        """Calcula energía y SOC tras P sin modificar el estado actual."""
        energy = self.soc * self.C
        if p_grid_kw >= 0.0:  # Convención interna legacy: positivo carga.
            energy_next = energy + self.eta_ch * p_grid_kw
        else:
            energy_next = energy + p_grid_kw / max(self.eta_dch, 1e-9)
        soc_next = float(np.clip(energy_next / self.C, self.soc_min, 1.0))
        return float(energy_next), soc_next

    def resolve_action_at(self, t: int, aP: float, aQ: float) -> V2GActionResolution:
        """Resuelve P/Q/SOC de ``[t,t+1)`` sin aplicar ninguna mutación."""
        t = self._assert_current_time(t)
        aP_requested = float(aP)
        aQ_requested = float(aQ)
        soc_before = float(self.soc)
        energy_before = soc_before * self.C

        if not self.connected:
            aP_applied = 0.0
            aQ_applied = 0.0
            p_grid_kw = 0.0
            q_grid_kvar = 0.0
            q_max_kvar = self.s_max if self.allow_q_when_disconnected else 0.0
            energy_next = energy_before
            soc_next = soc_before
        else:
            aP_applied = float(np.clip(aP_requested, -1.0, 1.0))
            aQ_applied = float(np.clip(aQ_requested, -1.0, 1.0))
            p_grid_kw = self._clip_p_by_soc(aP_applied * self.p_ch_max)
            q_max_kvar = float(np.sqrt(max(self.s_max**2 - p_grid_kw**2, 0.0)))
            q_grid_kvar = float(
                np.clip(aQ_applied * self.s_max, -q_max_kvar, q_max_kvar)
            )
            energy_next, soc_next = self._resolve_soc_after_grid_power(p_grid_kw)

        return V2GActionResolution(
            t=t,
            connected=bool(self.connected),
            aP_requested=aP_requested,
            aQ_requested=aQ_requested,
            aP_applied=aP_applied,
            aQ_applied=aQ_applied,
            P_grid_kw=float(p_grid_kw),
            Q_grid_kvar=float(q_grid_kvar),
            q_max_kvar=float(q_max_kvar),
            E_before_kWh=float(energy_before),
            E_next_kWh=float(energy_next),
            soc_before=soc_before,
            soc_next=float(soc_next),
        )

    def commit_action_at(self, t: int, resolution: V2GActionResolution) -> tuple[float, float]:
        """Confirma una resolución vigente sin avanzar el reloj del camión."""
        t = self._assert_current_time(t)
        if not isinstance(resolution, V2GActionResolution):
            raise TypeError("resolution debe ser un V2GActionResolution.")
        if resolution.t != t or resolution.connected != self.connected:
            raise RuntimeError("La resolución del truck no corresponde al estado actual.")
        if self._action_applied_t == t:
            raise RuntimeError(f"Ya se aplicó una acción del truck en t={t}.")
        if not np.isclose(resolution.soc_before, self.soc, rtol=0.0, atol=1e-9):
            raise RuntimeError("La resolución del truck está desactualizada.")

        self.soc = resolution.soc_next
        self.P_grid_kw = resolution.P_grid_kw
        self.Q_grid_kvar = resolution.Q_grid_kvar
        self.last_qmax = resolution.q_max_kvar
        self._action_applied_t = t
        return self.P_grid_kw, self.Q_grid_kvar

    def apply_action_at(self, t: int, aP: float, aQ: float) -> tuple[float, float]:
        """Aplica P/Q en ``[t,t+1)`` sin avanzar el reloj.

        ``aP>0`` conserva la convención interna de carga/absorción. El futuro
        decoder del ambiente invertirá P para la convención externa oficial.
        """
        resolution = self.resolve_action_at(t, aP, aQ)
        return self.commit_action_at(t, resolution)

    def advance_interval(self, t: int) -> tuple[float, bool]:
        """Cierra ``[t,t+1)`` y sincroniza el estado con ``t+1``."""
        t = self._assert_current_time(t)
        if self._action_applied_t != t:
            raise RuntimeError("Aplica exactamente una acción antes de avanzar el truck.")

        next_t = t + 1
        penalty_depart = 0.0
        if next_t < self.horizon_hours:
            next_connected = self.available_at(next_t)
            if self.connected and not next_connected:
                penalty_depart = max(0.0, self.soc_target_dep - self.soc)
            elif not self.connected and next_connected:
                self._sample_arrival_soc()
            self.connected = next_connected
            self.last_qmax = self.s_max if self.connected else (
                self.s_max if self.allow_q_when_disconnected else 0.0
            )
        else:
            self.connected = False
            self.last_qmax = 0.0

        self.t = next_t
        self._action_applied_t = None
        return float(penalty_depart), self.t >= self.horizon_hours

    def step(self, aP: float, aQ: float):
        """API legacy mutable; devuelve el estado sincronizado de ``t+1``."""
        t = self._assert_current_time(self.t)
        p_q = self.apply_action_at(t, aP, aQ)
        penalty_depart, done = self.advance_interval(t)
        state_next = self._terminal_state() if done else self.state_at(self.t)
        return state_next, p_q, penalty_depart, done


availability_bus5_7x24 = np.array([
    [1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=int)

availability_bus6_7x24 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=int)
