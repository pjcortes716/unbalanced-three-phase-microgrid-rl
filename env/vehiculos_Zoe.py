import operator

import numpy as np


class ZoeFleetBus3:
    """Dos Zoe monofásicos exógenos conectados en bus 3.

    El tiempo externo representa el inicio del intervalo ``[t, t+1)``. La
    interfaz nueva separa :meth:`prepare_interval` (eventos de frontera),
    :meth:`get_power_request` (consulta no mutante) y
    :meth:`advance_interval` (actualización energética). ``step`` se conserva
    como API legacy para notebooks existentes.
    """

    horizon_hours = 168

    def __init__(
        self,
        seed: int = 0,
        C_kwh: float = 52.0,
        Pmax_kw: float = 22.0,
        eta_ch: float = 0.95,
        soc_min: float = 0.10,
        soc_arr_min: float = 0.20,
        soc_arr_max: float = 0.40,
        soc_target_dep: float = 0.80,
        start_hour: int = 9,
        end_hour: int = 18,
    ):
        self.rng = np.random.default_rng(seed)
        self.C = float(C_kwh)
        self.Pmax = float(Pmax_kw)
        self.eta = float(eta_ch)
        self.soc_min = float(soc_min)
        self.soc_arr_min = float(soc_arr_min)
        self.soc_arr_max = float(soc_arr_max)
        self.soc_target = float(soc_target_dep)
        self.start_hour = int(start_hour)
        self.end_hour = int(end_hour)
        if not 0 <= self.start_hour < self.end_hour <= 24:
            raise ValueError("Se requiere 0 <= start_hour < end_hour <= 24.")

        self.soc = np.zeros(2, dtype=float)
        self.connected = np.zeros(2, dtype=bool)
        self.Eac_rem = np.zeros(2, dtype=float)
        self.t = 0  # Legacy: el ambiente futuro pasará t explícitamente.
        self._prepared_t: int | None = None

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

    def _weekday_and_hour(self, t: int) -> tuple[int, int]:
        t = self._validate_time(t)
        return t // 24, t % 24

    def _num_evs_present_today(self, day: int) -> int:
        if day <= 4:
            return 2
        if day == 5:
            return 1
        return 0

    def _arrival_event(self, n_present: int) -> None:
        self.connected[:] = False
        self.connected[:n_present] = True
        for i in range(2):
            if self.connected[i]:
                soc_arr = float(self.rng.uniform(self.soc_arr_min, self.soc_arr_max))
                self.soc[i] = max(soc_arr, self.soc_min)
                e_need_dc = max(0.0, (self.soc_target - self.soc[i]) * self.C)
                self.Eac_rem[i] = e_need_dc / self.eta
            else:
                self.Eac_rem[i] = 0.0

    def _departure_event(self) -> None:
        self.connected[:] = False
        self.Eac_rem[:] = 0.0

    def reset(self, *, rng=None, seed: int | None = None) -> None:
        """Reinicia el estado; admite RNG/seed explícito para reproducibilidad."""
        self._configure_rng(rng=rng, seed=seed)
        self.soc[:] = self.soc_target
        self.connected[:] = False
        self.Eac_rem[:] = 0.0
        self.t = 0
        self._prepared_t = None

    def prepare_interval(self, t: int) -> None:
        """Procesa llegadas/salidas de frontera sin entregar energía.

        Una llegada puede cambiar SOC/Eac_rem antes de ``obs_t`` porque es un
        evento físico de frontera. La consulta posterior de potencia no muta
        estado y esta preparación es idempotente para la misma hora.
        """
        t = self._validate_time(t)
        if t != self.t:
            raise ValueError(
                f"El estado Zoe está en t={self.t}; no se puede preparar t={t}. "
                "Avanza cada intervalo exactamente una vez."
            )
        if self._prepared_t == t:
            return
        if self._prepared_t is not None:
            raise RuntimeError("Debes avanzar el intervalo preparado antes de preparar otro.")

        day, hour = self._weekday_and_hour(t)
        n_present = self._num_evs_present_today(day)
        if hour == self.start_hour:
            self._arrival_event(n_present)
        elif hour == self.end_hour:
            self._departure_event()
        self._prepared_t = t

    def get_power_request(self, t: int) -> np.ndarray:
        """Devuelve P Zoe para ``[t,t+1)`` sin cambiar SOC, Eac_rem ni ``t``."""
        t = self._validate_time(t)
        if t != self.t or self._prepared_t != t:
            raise RuntimeError("Llama prepare_interval(t) antes de consultar la potencia Zoe.")

        day, hour = self._weekday_and_hour(t)
        in_window = self.start_hour <= hour < self.end_hour and self._num_evs_present_today(day) > 0
        if not in_window:
            return np.zeros(2, dtype=float)
        return np.where(self.connected, np.minimum(self.Pmax, self.Eac_rem), 0.0).astype(float, copy=True)

    def advance_interval(self, t: int, power_kw=None) -> bool:
        """Actualiza energía con la solicitud de t y deja el reloj en ``t+1``."""
        t = self._validate_time(t)
        requested = self.get_power_request(t)
        if power_kw is None:
            power_kw = requested
        power_kw = np.asarray(power_kw, dtype=float)
        if power_kw.shape != (2,):
            raise ValueError("power_kw debe tener shape (2,).")
        if not np.allclose(power_kw, requested, rtol=0.0, atol=1e-12):
            raise ValueError("power_kw debe coincidir con get_power_request(t).")

        for i, p_i in enumerate(requested):
            eac = float(p_i)  # dt_h = 1 h
            self.Eac_rem[i] = max(0.0, self.Eac_rem[i] - eac)
            self.soc[i] = min(self.soc_target, self.soc[i] + self.eta * eac / self.C)

        self.t += 1
        self._prepared_t = None
        return self.t >= self.horizon_hours

    def step(self):
        """API legacy mutable: una consulta y un avance por hora."""
        self.prepare_interval(self.t)
        power_kw = self.get_power_request(self.t)
        done = self.advance_interval(self.t, power_kw)
        return float(power_kw[0]), float(power_kw[1]), done
