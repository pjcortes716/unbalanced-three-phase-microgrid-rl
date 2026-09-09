"""Núcleo puro para eventos extremos de evaluación OOD.

El módulo no conoce ``MicrorredIndustrial`` ni ninguno de sus componentes.
Un ``ExtremeEventSpec`` describe un único evento inmutable y
``ExtremeEventManager`` transforma realizaciones ya calculadas, sin guardar
estado temporal ni consumir aleatoriedad. La integración futura deberá pasar
las realizaciones nominales a este módulo exactamente una vez por hora.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Mapping

import numpy as np


HORIZON_HOURS = 168
SOLAR_EPSILON = 1e-6
PHASE_ZERO_TOLERANCE = 1e-12

EVENT_TYPES = ("load_surge", "pv_collapse", "phase_imbalance", "compound")
RANDOM_EVENT_TYPE = "random"
LOAD_TARGETS = ("office", "wh1", "wh2")
PV_TARGETS = ("pv1", "pv2", "pv3")
PHASES = ("A", "B", "C")


def _validate_time(value: int, *, name: str, upper_bound: int = HORIZON_HOURS) -> int:
    """Valida un índice entero de hora dentro de ``[0, upper_bound)``."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} debe ser un entero, no bool.")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} debe ser un entero.") from exc
    if not 0 <= value < upper_bound:
        raise ValueError(f"{name}={value} debe estar en [0, {upper_bound - 1}].")
    return int(value)


def _validate_horizon(horizon: int) -> int:
    """Valida el horizonte opcional del generador sin exceder la semana."""
    if isinstance(horizon, (bool, np.bool_)):
        raise TypeError("horizon debe ser un entero, no bool.")
    try:
        horizon = operator.index(horizon)
    except TypeError as exc:
        raise TypeError("horizon debe ser un entero.") from exc
    if not 1 <= horizon <= HORIZON_HOURS:
        raise ValueError(f"horizon debe estar en [1, {HORIZON_HOURS}].")
    return int(horizon)


def _finite_float(value: float | None, *, name: str, required: bool) -> float | None:
    """Convierte un escalar a float finito, respetando opcionalidad."""
    if value is None:
        if required:
            raise ValueError(f"{name} es obligatorio.")
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} debe ser un escalar real, no bool.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} debe ser un escalar real finito.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} debe ser finito.")
    return result


def _validate_targets(
    targets: tuple[str, ...],
    *,
    name: str,
    allowed: tuple[str, ...],
) -> None:
    """Comprueba que una tupla de objetivos sea inmutable, única y conocida."""
    if not isinstance(targets, tuple):
        raise TypeError(f"{name} debe ser una tupla inmutable.")
    if not all(isinstance(target, str) for target in targets):
        raise TypeError(f"{name} debe contener únicamente strings.")
    unknown = set(targets) - set(allowed)
    if unknown:
        raise ValueError(f"{name} contiene targets desconocidos: {sorted(unknown)}.")
    if len(set(targets)) != len(targets):
        raise ValueError(f"{name} no puede contener targets repetidos.")


@dataclass(frozen=True)
class ExtremeEventSpec:
    """Descripción inmutable de un único evento en la semana ``[0, 168)``.

    ``end_t`` es exclusivo. Los rangos de severidad de esta clase son físicos
    y lógicos, no los rangos por defecto del generador: se admiten escenarios
    personalizados más severos mientras sigan siendo físicamente coherentes.
    """

    event_type: str
    start_t: int
    end_t: int
    load_targets: tuple[str, ...] = ()
    pv_targets: tuple[str, ...] = ()
    load_multiplier: float | None = None
    pv_alpha: float | None = None
    dominant_phase: str | None = None
    phase_share: float | None = None

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(
                f"event_type debe ser uno de {EVENT_TYPES}; 'random' no es un spec final."
            )
        if isinstance(self.start_t, (bool, np.bool_)) or isinstance(self.end_t, (bool, np.bool_)):
            raise TypeError("start_t y end_t deben ser enteros, no bool.")
        try:
            start_t = operator.index(self.start_t)
            end_t = operator.index(self.end_t)
        except TypeError as exc:
            raise TypeError("start_t y end_t deben ser enteros.") from exc
        if not 0 <= start_t < end_t <= HORIZON_HOURS:
            raise ValueError(f"Se requiere 0 <= start_t < end_t <= {HORIZON_HOURS}.")

        _validate_targets(self.load_targets, name="load_targets", allowed=LOAD_TARGETS)
        _validate_targets(self.pv_targets, name="pv_targets", allowed=PV_TARGETS)

        load_multiplier = _finite_float(
            self.load_multiplier,
            name="load_multiplier",
            required=self.event_type in {"load_surge", "compound"},
        )
        pv_alpha = _finite_float(
            self.pv_alpha,
            name="pv_alpha",
            required=self.event_type in {"pv_collapse", "compound"},
        )
        phase_share = _finite_float(
            self.phase_share,
            name="phase_share",
            required=self.event_type == "phase_imbalance",
        )

        if load_multiplier is not None and load_multiplier < 1.0:
            raise ValueError("load_multiplier debe ser mayor o igual que 1.0.")
        if pv_alpha is not None and not 0.0 <= pv_alpha <= 1.0:
            raise ValueError("pv_alpha debe estar en [0, 1].")
        if phase_share is not None and not (1.0 / 3.0) <= phase_share <= 1.0:
            raise ValueError("phase_share debe estar en [1/3, 1].")

        if self.dominant_phase is not None and self.dominant_phase not in PHASES:
            raise ValueError(f"dominant_phase debe ser uno de {PHASES} o None.")

        if self.event_type == "load_surge":
            if not self.load_targets:
                raise ValueError("load_surge requiere al menos un load_target.")
            if self.pv_targets or pv_alpha is not None or self.dominant_phase is not None or phase_share is not None:
                raise ValueError("load_surge no admite campos PV ni de phase_imbalance.")
        elif self.event_type == "pv_collapse":
            if not self.pv_targets:
                raise ValueError("pv_collapse requiere al menos un pv_target.")
            if self.load_targets or load_multiplier is not None or self.dominant_phase is not None or phase_share is not None:
                raise ValueError("pv_collapse no admite campos de carga ni de phase_imbalance.")
        elif self.event_type == "phase_imbalance":
            if len(self.load_targets) != 1:
                raise ValueError("phase_imbalance requiere exactamente un load_target.")
            if self.pv_targets or load_multiplier is not None or pv_alpha is not None:
                raise ValueError("phase_imbalance no admite campos de surge ni PV.")
            if self.dominant_phase is None:
                raise ValueError("phase_imbalance requiere dominant_phase.")
        elif self.event_type == "compound":
            if not self.load_targets or not self.pv_targets:
                raise ValueError("compound requiere load_targets y pv_targets no vacíos.")
            if self.dominant_phase is not None or phase_share is not None:
                raise ValueError("compound no admite campos de phase_imbalance.")

    @property
    def duration(self) -> int:
        """Duración entera del evento en horas."""
        return int(self.end_t - self.start_t)


class ExtremeEventManager:
    """Aplica un spec sin consumir RNG ni guardar evolución temporal.

    ``adjust_load_abc`` recibe dos vectores de tres fases y devuelve copias.
    ``adjust_pv_available`` recibe un mapping ``{"pv1": kW, ...}`` y devuelve
    un ``dict`` nuevo. ``pv_collapse`` representa irradiancia: solo modifica
    disponibilidad activa, nunca capacidad reactiva o estado de inversores.
    """

    __slots__ = ("spec",)

    def __init__(self, spec: ExtremeEventSpec) -> None:
        if not isinstance(spec, ExtremeEventSpec):
            raise TypeError("spec debe ser una instancia de ExtremeEventSpec.")
        self.spec = spec

    @staticmethod
    def _validate_query_time(t: int) -> int:
        return _validate_time(t, name="t")

    def is_active(self, t: int) -> bool:
        """Indica si ``t`` pertenece a la ventana semiabierta del evento."""
        t = self._validate_query_time(t)
        return bool(self.spec.start_t <= t < self.spec.end_t)

    @staticmethod
    def _three_phase_copy(values: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape != (3,):
            raise ValueError(f"{name} debe tener shape (3,), no {array.shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} debe contener valores finitos.")
        return array.copy()

    @staticmethod
    def _validate_load_target(target: str) -> None:
        if target not in LOAD_TARGETS:
            raise ValueError(f"target de carga desconocido: {target!r}.")

    def _phase_shares(self, p_abc: np.ndarray, q_abc: np.ndarray) -> np.ndarray:
        """Construye el reparto ABC para phase_imbalance sin mutar entradas."""
        assert self.spec.event_type == "phase_imbalance"
        assert self.spec.dominant_phase is not None
        assert self.spec.phase_share is not None

        p_total = float(np.sum(p_abc))
        q_total = float(np.sum(q_abc))
        if abs(p_total) <= PHASE_ZERO_TOLERANCE and abs(q_total) <= PHASE_ZERO_TOLERANCE:
            return np.full(3, 1.0 / 3.0, dtype=float)

        dominant_idx = PHASES.index(self.spec.dominant_phase)
        other_indices = tuple(index for index in range(3) if index != dominant_idx)
        other_values = p_abc[list(other_indices)]
        other_sum = float(np.sum(other_values))
        if abs(other_sum) <= PHASE_ZERO_TOLERANCE:
            other_values = q_abc[list(other_indices)]
            other_sum = float(np.sum(other_values))

        if abs(other_sum) <= PHASE_ZERO_TOLERANCE:
            other_ratios = np.array([0.5, 0.5], dtype=float)
        else:
            other_ratios = np.asarray(other_values / other_sum, dtype=float)

        shares = np.zeros(3, dtype=float)
        shares[dominant_idx] = self.spec.phase_share
        shares[list(other_indices)] = (1.0 - self.spec.phase_share) * other_ratios
        return shares

    def adjust_load_abc(
        self,
        t: int,
        target: str,
        p_abc: np.ndarray,
        q_abc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve P/Q ABC efectivo para una carga, sin alterar sus entradas."""
        active = self.is_active(t)
        self._validate_load_target(target)
        p_new = self._three_phase_copy(p_abc, name="p_abc")
        q_new = self._three_phase_copy(q_abc, name="q_abc")
        if not active:
            return p_new, q_new

        if self.spec.event_type in {"load_surge", "compound"} and target in self.spec.load_targets:
            assert self.spec.load_multiplier is not None
            return p_new * self.spec.load_multiplier, q_new * self.spec.load_multiplier

        if self.spec.event_type != "phase_imbalance" or target not in self.spec.load_targets:
            return p_new, q_new

        p_total = float(np.sum(p_new))
        q_total = float(np.sum(q_new))
        if abs(p_total) <= PHASE_ZERO_TOLERANCE and abs(q_total) <= PHASE_ZERO_TOLERANCE:
            return p_new, q_new

        shares = self._phase_shares(p_new, q_new)
        p_result = p_total * shares
        q_result = q_total * shares
        # Una fase no dominante absorbe el redondeo. Así se conserva la
        # fracción dominante incluso cuando la fase C es la dominante.
        assert self.spec.dominant_phase is not None
        dominant_idx = PHASES.index(self.spec.dominant_phase)
        other_indices = tuple(index for index in range(3) if index != dominant_idx)
        residual_idx = other_indices[1]
        fixed_idx = other_indices[0]
        p_result[residual_idx] = p_total - p_result[dominant_idx] - p_result[fixed_idx]
        q_result[residual_idx] = q_total - q_result[dominant_idx] - q_result[fixed_idx]
        return p_result, q_result

    def adjust_pv_available(self, t: int, pv_available: Mapping[str, float]) -> dict[str, float]:
        """Devuelve disponibilidades activas FV efectivas en un mapping nuevo."""
        active = self.is_active(t)
        if not isinstance(pv_available, Mapping):
            raise TypeError("pv_available debe ser un mapping {'pv1': kW, ...}.")

        result: dict[str, float] = {}
        for target, value in pv_available.items():
            if target not in PV_TARGETS:
                raise ValueError(f"target PV desconocido: {target!r}.")
            value = _finite_float(value, name=f"pv_available[{target!r}]", required=True)
            assert value is not None
            if value < 0.0:
                raise ValueError("La disponibilidad PV no puede ser negativa.")
            result[target] = value

        if active and self.spec.event_type in {"pv_collapse", "compound"}:
            assert self.spec.pv_alpha is not None
            for target in self.spec.pv_targets:
                if target in result:
                    result[target] = self.spec.pv_alpha * result[target]
        return result

    def metadata_at(self, t: int) -> dict[str, object]:
        """Genera metadata serializable para un futuro ``info`` del ambiente."""
        active = self.is_active(t)
        if self.spec.event_type == "load_surge":
            intensity: dict[str, float] = {"load_multiplier": float(self.spec.load_multiplier)}
        elif self.spec.event_type == "pv_collapse":
            intensity = {"pv_alpha": float(self.spec.pv_alpha)}
        elif self.spec.event_type == "phase_imbalance":
            intensity = {"phase_share": float(self.spec.phase_share)}
        else:
            intensity = {
                "load_multiplier": float(self.spec.load_multiplier),
                "pv_alpha": float(self.spec.pv_alpha),
            }
        return {
            "extreme_event_enabled": True,
            "extreme_event_active": active,
            "extreme_event_type": self.spec.event_type,
            "extreme_event_start": int(self.spec.start_t),
            "extreme_event_end": int(self.spec.end_t),
            "extreme_event_targets": {
                "load_targets": list(self.spec.load_targets),
                "pv_targets": list(self.spec.pv_targets),
            },
            "extreme_event_intensity": intensity,
            "extreme_event_phase": self.spec.dominant_phase,
        }


def _require_rng(rng: object) -> None:
    """Comprueba la API mínima del RNG recibido explícitamente."""
    if not callable(getattr(rng, "integers", None)) or not callable(getattr(rng, "uniform", None)):
        raise TypeError("rng debe exponer los métodos integers() y uniform().")


def _sample_duration(rng: np.random.Generator, lower: int, upper: int) -> int:
    return int(rng.integers(lower, upper + 1))


def _sample_start(rng: np.random.Generator, *, horizon: int, duration: int) -> int:
    return int(rng.integers(0, horizon - duration + 1))


def _solar_start_candidates(
    solar_base: np.ndarray | None,
    *,
    horizon: int,
    duration: int,
    epsilon: float,
) -> np.ndarray:
    """Encuentra inicios cuya ventana completa tiene irradiancia positiva."""
    if solar_base is None:
        raise ValueError("solar_base es obligatorio para eventos PV diurnos.")
    solar = np.asarray(solar_base, dtype=float)
    if solar.shape != (horizon,):
        raise ValueError(f"solar_base debe tener shape ({horizon},), no {solar.shape}.")
    if not np.all(np.isfinite(solar)):
        raise ValueError("solar_base debe contener valores finitos.")
    return np.array(
        [
            start_t
            for start_t in range(horizon - duration + 1)
            if np.all(solar[start_t : start_t + duration] > epsilon)
        ],
        dtype=int,
    )


def _sample_daylight_start(
    rng: np.random.Generator,
    *,
    solar_base: np.ndarray | None,
    horizon: int,
    duration: int,
    epsilon: float,
) -> int:
    candidates = _solar_start_candidates(
        solar_base,
        horizon=horizon,
        duration=duration,
        epsilon=epsilon,
    )
    if candidates.size == 0:
        raise ValueError(
            "No existe una ventana solar diurna válida para la duración solicitada."
        )
    return int(candidates[int(rng.integers(0, candidates.size))])


def generate_extreme_event_spec(
    event_type: str,
    rng: np.random.Generator,
    *,
    horizon: int = HORIZON_HOURS,
    solar_base: np.ndarray | None = None,
    solar_epsilon: float = SOLAR_EPSILON,
) -> ExtremeEventSpec:
    """Genera un spec concreto usando exclusivamente el RNG proporcionado.

    Los eventos PV y compound requieren una ventana completa de
    ``solar_base > solar_epsilon``. La duración se muestrea una sola vez: si
    no existe una ventana para esa duración, se informa el error en lugar de
    reducirla o cambiar el tipo de evento.
    """
    _require_rng(rng)
    horizon = _validate_horizon(horizon)
    epsilon = _finite_float(solar_epsilon, name="solar_epsilon", required=True)
    assert epsilon is not None
    if epsilon < 0.0:
        raise ValueError("solar_epsilon debe ser mayor o igual que cero.")

    if event_type == RANDOM_EVENT_TYPE:
        event_type = EVENT_TYPES[int(rng.integers(0, len(EVENT_TYPES)))]
    if event_type not in EVENT_TYPES:
        raise ValueError(f"event_type debe ser uno de {EVENT_TYPES} o 'random'.")

    if event_type == "load_surge":
        duration = _sample_duration(rng, 2, 6)
        if duration > horizon:
            raise ValueError("horizon no admite la duración mínima de load_surge.")
        start_t = _sample_start(rng, horizon=horizon, duration=duration)
        return ExtremeEventSpec(
            event_type=event_type,
            start_t=start_t,
            end_t=start_t + duration,
            load_targets=(LOAD_TARGETS[int(rng.integers(0, len(LOAD_TARGETS)))],),
            load_multiplier=float(rng.uniform(1.35, 1.70)),
        )

    if event_type == "phase_imbalance":
        duration = _sample_duration(rng, 2, 4)
        if duration > horizon:
            raise ValueError("horizon no admite la duración mínima de phase_imbalance.")
        start_t = _sample_start(rng, horizon=horizon, duration=duration)
        return ExtremeEventSpec(
            event_type=event_type,
            start_t=start_t,
            end_t=start_t + duration,
            load_targets=(LOAD_TARGETS[int(rng.integers(0, len(LOAD_TARGETS)))],),
            dominant_phase=PHASES[int(rng.integers(0, len(PHASES)))],
            phase_share=float(rng.uniform(0.55, 0.70)),
        )

    if event_type == "pv_collapse":
        duration = _sample_duration(rng, 2, 6)
        if duration > horizon:
            raise ValueError("horizon no admite la duración mínima de pv_collapse.")
        start_t = _sample_daylight_start(
            rng,
            solar_base=solar_base,
            horizon=horizon,
            duration=duration,
            epsilon=epsilon,
        )
        return ExtremeEventSpec(
            event_type=event_type,
            start_t=start_t,
            end_t=start_t + duration,
            pv_targets=PV_TARGETS,
            pv_alpha=float(rng.uniform(0.0, 0.30)),
        )

    # compound combina exclusivamente surge de carga y caída de irradiancia.
    duration = _sample_duration(rng, 2, 5)
    if duration > horizon:
        raise ValueError("horizon no admite la duración mínima de compound.")
    start_t = _sample_daylight_start(
        rng,
        solar_base=solar_base,
        horizon=horizon,
        duration=duration,
        epsilon=epsilon,
    )
    return ExtremeEventSpec(
        event_type="compound",
        start_t=start_t,
        end_t=start_t + duration,
        load_targets=(LOAD_TARGETS[int(rng.integers(0, len(LOAD_TARGETS)))],),
        pv_targets=PV_TARGETS,
        load_multiplier=float(rng.uniform(1.35, 1.60)),
        pv_alpha=float(rng.uniform(0.0, 0.25)),
    )
