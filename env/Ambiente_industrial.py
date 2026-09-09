import math
import operator
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

from .battery import Battery, BatteryParams
from .oficinas import OfficeLoadSampler
from .PV_plant import WeeklyPVSampler, PVPlant
from .price_sampler import PRICE_PROFILE_24H, PRICE_SCALE, PriceSampler
from .v2g_bus_5 import V2GTruck
from .vehiculos_Zoe import ZoeFleetBus3
from .opendss_easy import DSSCircuit 
from .extreme_events import ExtremeEventManager, ExtremeEventSpec, generate_extreme_event_spec
from .scenario_catalog import ScenarioCatalog
from .benchmark_metrics import EpisodeMetrics, truck_departure_event


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_LABELS = ("A", "B", "C")
_POWER_FACTOR_EPSILON = 1e-9
_VOLTAGE_SEQUENCE_EPSILON = 1e-12
_PF_INPUT_TOLERANCE = 1e-9
# Umbral de diseño del ambiente. Se evalúa separadamente en A, B y C.
PF_MIN = 0.90
# El perfil de PriceSampler está expresado en centavos USD/kWh. Esta
# conversión se usa solamente para la contabilidad económica de la reward;
# no altera el precio crudo de la observación.
PRICE_RAW_TO_USD_PER_KWH = 1.0 / 100.0
DATA_DIR = PROJECT_ROOT / "data" / "train"


def _data_file(filename: str) -> Path:
    """Devuelve un pool canónico sin depender del directorio de trabajo."""
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos canónico '{filename}' en '{DATA_DIR}'. "
            f"Ruta esperada: '{path}'."
        )
    return path


def calculate_voltage_unbalance_pct(
    voltage_phasors_abc: np.ndarray,
    *,
    epsilon: float = _VOLTAGE_SEQUENCE_EPSILON,
) -> float:
    """Calcula el VUF como ``100 * |V2| / |V1|`` a partir de fasores ABC.

    El orden de entrada es estrictamente ``[A, B, C]``. ``V1`` y ``V2`` son,
    respectivamente, las componentes simétricas de secuencia positiva y
    negativa, con ``a = exp(j*2*pi/3)``. Por tanto este valor es el cociente
    *negative-sequence / positive-sequence voltage ratio*, no una aproximación
    basada en los módulos de tensión.
    """
    phasors = np.asarray(voltage_phasors_abc, dtype=np.complex128)
    if phasors.shape != (3,) or not np.all(np.isfinite(phasors)):
        raise ValueError("voltage_phasors_abc debe contener tres fasores ABC finitos.")

    a = np.exp(2j * np.pi / 3.0)
    v1 = (phasors[0] + a * phasors[1] + a**2 * phasors[2]) / 3.0
    v2 = (phasors[0] + a**2 * phasors[1] + a * phasors[2]) / 3.0
    if abs(v1) <= epsilon:
        raise ValueError("La secuencia positiva es demasiado pequeña para calcular VUF.")
    return float(100.0 * abs(v2) / abs(v1))


def calculate_power_factor(
    p_kw: float,
    q_kvar: float,
    *,
    epsilon: float = _POWER_FACTOR_EPSILON,
) -> float:
    """Devuelve ``|P| / hypot(P, Q)`` en ``[0, 1]`` de forma robusta.

    Para potencia aparente prácticamente nula (P y Q próximos a cero) se
    devuelve 1.0. Si P es aproximadamente cero y Q no lo es, la fórmula
    devuelve aproximadamente 0.0. El signo de P/Q no se codifica en el PF.
    """
    p_kw = float(p_kw)
    q_kvar = float(q_kvar)
    if not (math.isfinite(p_kw) and math.isfinite(q_kvar)):
        raise ValueError("P y Q deben ser finitos para calcular el factor de potencia.")
    apparent_power = math.hypot(p_kw, q_kvar)
    if apparent_power <= epsilon:
        return 1.0
    return float(np.clip(abs(p_kw) / apparent_power, 0.0, 1.0))


@dataclass(frozen=True)
class RewardBreakdown:
    """Desglose inmutable de la reward nominal de un intervalo.

    Los precios crudos de ``PriceSampler`` son centavos de USD/kWh, mientras
    que los costos ``energy_cost_usd`` (alias ``operating_cost_usd``),
    ``pf_penalty_usd``, ``truck_departure_penalty_usd`` y ``reward`` están en USD.
    ``lambda_pf`` se interpreta como USD/(kVAr h). Los vectores siguen
    estrictamente el orden ``[A, B, C]``.
    """

    reward: float
    operating_cost_usd: float
    pf_penalty_usd: float
    price_raw: float
    price_usd_per_kwh: float
    p_grid_kw: float
    pf_phase: np.ndarray
    pf_violation_phase: np.ndarray
    q_pcc_kvar_phase: np.ndarray
    dt_h: float
    lambda_pf: float
    truck_departure_penalty_usd: float = 0.0

    @property
    def energy_cost_usd(self) -> float:
        """Economic energy term; operating_cost_usd is retained as a legacy alias."""
        return self.operating_cost_usd


def _validate_reward_scalar(value: float, *, name: str) -> float:
    """Convierte y valida un escalar finito usado por :func:`compute_reward`."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} debe ser un escalar finito.") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} debe ser un escalar finito.")
    return scalar


def _validate_reward_phase_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    """Devuelve un vector ABC finito de forma estrictamente ``(3,)``."""
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} debe ser un vector numérico finito con shape (3,).") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} debe ser un vector finito con shape (3,).")
    return vector.copy()


def compute_reward(
    *,
    p_grid_kw: float,
    price_raw: float,
    pf_phase: np.ndarray,
    q_pcc_kvar_phase: np.ndarray,
    dt_h: float,
    lambda_pf: float,
    truck_departure_penalty_usd: float = 0.0,
) -> RewardBreakdown:
    """Calcula la reward nominal sin consultar ni mutar el ambiente.

    La definición implementada es::

        operating_cost_usd = P_grid_kw * (price_raw / 100) * dt_h
        pf_violation_phi = max(0, PF_MIN - PF_phi)
        pf_penalty_usd = lambda_pf * dt_h * sum(
            pf_violation_phi * abs(Q_pcc_phi_kvar)
        )
        reward = -operating_cost_usd - pf_penalty_usd - truck_departure_penalty_usd

    ``P_grid_kw`` positivo representa importación y negativo representa
    exportación. PF y Q deben proceder en el futuro directamente de
    ``ElectricalSnapshotResult.pcc_pf_phase`` y
    ``ElectricalSnapshotResult.pcc_q_kvar_phase``; esta función no vuelve a
    calcular el factor de potencia. ``lambda_pf`` es un parámetro de modelo
    configurable externamente, no un hiperparámetro interno de RL.
    """
    p_grid_kw = _validate_reward_scalar(p_grid_kw, name="p_grid_kw")
    price_raw = _validate_reward_scalar(price_raw, name="price_raw")
    dt_h = _validate_reward_scalar(dt_h, name="dt_h")
    lambda_pf = _validate_reward_scalar(lambda_pf, name="lambda_pf")
    truck_departure_penalty_usd = _validate_reward_scalar(
        truck_departure_penalty_usd, name="truck_departure_penalty_usd"
    )
    if truck_departure_penalty_usd < 0:
        raise ValueError("truck_departure_penalty_usd debe ser mayor o igual que cero.")
    if dt_h <= 0.0:
        raise ValueError("dt_h debe ser mayor que cero.")
    if lambda_pf < 0.0:
        raise ValueError("lambda_pf debe ser mayor o igual que cero.")

    pf_phase = _validate_reward_phase_vector(pf_phase, name="pf_phase")
    q_pcc_kvar_phase = _validate_reward_phase_vector(
        q_pcc_kvar_phase,
        name="q_pcc_kvar_phase",
    )
    if np.any(pf_phase < -_PF_INPUT_TOLERANCE) or np.any(
        pf_phase > 1.0 + _PF_INPUT_TOLERANCE
    ):
        raise ValueError("pf_phase debe estar en el intervalo [0, 1].")
    # Se toleran únicamente desviaciones de redondeo alrededor de los límites.
    pf_phase = np.clip(pf_phase, 0.0, 1.0)

    price_usd_per_kwh = price_raw * PRICE_RAW_TO_USD_PER_KWH
    operating_cost_usd = p_grid_kw * price_usd_per_kwh * dt_h
    pf_violation_phase = np.maximum(0.0, PF_MIN - pf_phase)
    pf_penalty_usd = lambda_pf * dt_h * float(
        np.sum(pf_violation_phase * np.abs(q_pcc_kvar_phase))
    )
    reward = -operating_cost_usd - pf_penalty_usd - truck_departure_penalty_usd

    for vector in (pf_phase, pf_violation_phase, q_pcc_kvar_phase):
        vector.setflags(write=False)
    return RewardBreakdown(
        reward=float(reward),
        operating_cost_usd=float(operating_cost_usd),
        pf_penalty_usd=float(pf_penalty_usd),
        price_raw=float(price_raw),
        price_usd_per_kwh=float(price_usd_per_kwh),
        p_grid_kw=float(p_grid_kw),
        pf_phase=pf_phase,
        pf_violation_phase=pf_violation_phase,
        q_pcc_kvar_phase=q_pcc_kvar_phase,
        dt_h=float(dt_h),
        lambda_pf=float(lambda_pf),
        truck_departure_penalty_usd=float(truck_departure_penalty_usd),
    )


@dataclass(frozen=True)
class ElectricalSnapshotInputs:
    """Potencias físicas realizadas durante una hora para la red OpenDSS.

    Las cargas de edificio se expresan por fase ``[A, B, C]`` y son positivas
    al consumir. Las potencias de activos controlables siguen la convención
    física del ambiente: P/Q positivos inyectan hacia la microrred.

    Este objeto no contiene una acción ni un instante de tiempo a propósito:
    el snapshot eléctrico no decide controles ni avanza la dinámica.
    """

    p_bus3_abc_kw: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    q_bus3_abc_kvar: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    p_wh1_abc_kw: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    q_wh1_abc_kvar: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    p_wh2_abc_kw: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    q_wh2_abc_kvar: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    pv1_p_inj_kw: float = 0.0
    pv1_q_inj_kvar: float = 0.0
    pv2_p_inj_kw: float = 0.0
    pv2_q_inj_kvar: float = 0.0
    pv3_p_inj_kw: float = 0.0
    pv3_q_inj_kvar: float = 0.0
    bess_p_inj_kw: float = 0.0
    bess_q_inj_kvar: float = 0.0
    truck5_p_inj_kw: float = 0.0
    truck5_q_inj_kvar: float = 0.0
    truck6_p_inj_kw: float = 0.0
    truck6_q_inj_kvar: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "p_bus3_abc_kw", "q_bus3_abc_kvar",
            "p_wh1_abc_kw", "q_wh1_abc_kvar",
            "p_wh2_abc_kw", "q_wh2_abc_kvar",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,):
                raise ValueError(f"{name} debe tener shape (3,), no {values.shape}.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} debe contener valores finitos.")
            object.__setattr__(self, name, values.copy())

        for name in (
            "pv1_p_inj_kw", "pv1_q_inj_kvar", "pv2_p_inj_kw", "pv2_q_inj_kvar",
            "pv3_p_inj_kw", "pv3_q_inj_kvar", "bess_p_inj_kw", "bess_q_inj_kvar",
            "truck5_p_inj_kw", "truck5_q_inj_kvar", "truck6_p_inj_kw",
            "truck6_q_inj_kvar",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} debe ser finito.")
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ElectricalSnapshotInputs":
        """Construye inputs desde el layout plano o de activos anidados.

        Además de los campos del dataclass, se admite por claridad, por
        ejemplo, ``{"pv1": {"p_inj_kw": 15, "q_inj_kvar": 0}}``.
        """
        flat = dict(values)
        for asset in ("pv1", "pv2", "pv3", "bess", "truck5", "truck6"):
            nested = flat.pop(asset, None)
            if nested is None:
                continue
            if not isinstance(nested, Mapping):
                raise ValueError(f"{asset} debe ser un mapping con p_inj_kw y q_inj_kvar.")
            unknown = set(nested) - {"p_inj_kw", "q_inj_kvar"}
            if unknown:
                raise ValueError(f"Campos desconocidos para {asset}: {sorted(unknown)}.")
            flat.setdefault(f"{asset}_p_inj_kw", nested.get("p_inj_kw", 0.0))
            flat.setdefault(f"{asset}_q_inj_kvar", nested.get("q_inj_kvar", 0.0))
        return cls(**flat)


@dataclass(frozen=True)
class ElectricalSnapshotResult:
    """Resultado estacionario de OpenDSS con diagnósticos eléctricos ABC.

    Las métricas por fase respetan siempre el orden ``[A, B, C]``. Los fasores
    de tensión proceden de las magnitudes y ángulos que resolvió OpenDSS; los
    diagnósticos aquí contenidos no forman parte de la observación RL.
    """

    converged: bool
    pcc: dict[str, float] | None
    voltages_vpu: dict[str, np.ndarray] | None
    voltage_angles_deg: dict[str, np.ndarray] | None
    voltage_phasors_v: dict[str, np.ndarray] | None
    voltage_unbalance_pct: dict[str, float] | None
    max_voltage_unbalance_pct: float | None
    worst_unbalance_bus: str | None
    max_lv_voltage_unbalance_pct: float | None
    pcc_p_kw_phase: np.ndarray | None
    pcc_q_kvar_phase: np.ndarray | None
    pcc_pf_phase: np.ndarray | None
    pcc_pf_total: float | None
    vmin_pu: float | None
    vmax_pu: float | None
    vmin_bus: str | None
    vmin_phase: str | None
    vmax_bus: str | None
    vmax_phase: str | None
    p_losses_kw: float | None
    q_losses_kvar: float | None
    power_balance: dict[str, Any] | None
    balance_residual_kw: float | None
    element_powers: dict[str, dict[str, np.ndarray]] | None
    diagnostics: dict[str, Any]
    bus2_vpu: np.ndarray | None
    transformer_loading_percent: float | None
    transformer_primary_p_kw: float | None
    transformer_primary_q_kvar: float | None
    transformer_secondary_p_kw: float | None
    transformer_secondary_q_kvar: float | None
    transformer_p_losses_kw: float | None
    transformer_q_losses_kvar: float | None

    def electrical_info(self) -> dict[str, Any]:
        """Convierte los diagnósticos del snapshot en un ``info`` serializable.

        Este helper no resuelve de nuevo la red ni muta el ambiente. Los arrays
        se convierten a listas de ``float`` para que un futuro ``step()`` pueda
        hacer ``info.update(snapshot.electrical_info())`` sin duplicar lógica.
        """
        info: dict[str, Any] = {"electrical_converged": bool(self.converged)}
        if not self.converged:
            return info

        required_values = (
            self.pcc,
            self.voltages_vpu,
            self.voltage_angles_deg,
            self.voltage_unbalance_pct,
            self.max_voltage_unbalance_pct,
            self.worst_unbalance_bus,
            self.max_lv_voltage_unbalance_pct,
            self.pcc_p_kw_phase,
            self.pcc_q_kvar_phase,
            self.pcc_pf_phase,
            self.pcc_pf_total,
            self.vmin_pu,
            self.vmax_pu,
            self.vmin_bus,
            self.vmin_phase,
            self.vmax_bus,
            self.vmax_phase,
            self.transformer_loading_percent,
        )
        if any(value is None for value in required_values):
            raise RuntimeError("El snapshot convergente no contiene toda la instrumentación eléctrica.")

        return {
            **info,
            "voltage_phase_order": list(PHASE_LABELS),
            "voltages_pu": {
                bus: [float(value) for value in values]
                for bus, values in self.voltages_vpu.items()
            },
            "voltage_angles_deg": {
                bus: [float(value) for value in values]
                for bus, values in self.voltage_angles_deg.items()
            },
            "voltage_unbalance_pct": {
                bus: float(value) for bus, value in self.voltage_unbalance_pct.items()
            },
            "max_voltage_unbalance_pct": float(self.max_voltage_unbalance_pct),
            "worst_unbalance_bus": self.worst_unbalance_bus,
            "max_lv_voltage_unbalance_pct": float(self.max_lv_voltage_unbalance_pct),
            "pcc_p_kw_phase": [float(value) for value in self.pcc_p_kw_phase],
            "pcc_q_kvar_phase": [float(value) for value in self.pcc_q_kvar_phase],
            "pcc_pf_phase": [float(value) for value in self.pcc_pf_phase],
            "pcc_p_kw": float(self.pcc["P_total_kW"]),
            "pcc_q_kvar": float(self.pcc["Q_total_kvar"]),
            "pcc_pf": float(self.pcc_pf_total),
            "transformer_loading_pct": float(self.transformer_loading_percent),
            "p_losses_kw": self.p_losses_kw,
            "q_losses_kvar": self.q_losses_kvar,
            "balance_residual_kw": self.balance_residual_kw,
            "vmin_pu": float(self.vmin_pu),
            "vmax_pu": float(self.vmax_pu),
            "vmin_bus": self.vmin_bus,
            "vmin_phase": self.vmin_phase,
            "vmax_bus": self.vmax_bus,
            "vmax_phase": self.vmax_phase,
        }


class MicrorredIndustrial(gym.Env):
    """
    Entorno Gymnasium de la microrred industrial trifásica desbalanceada.

    Por ahora esta clase solo define:
    - módulos exógenos (cargas, PV, EVs)
    - activos controlables (BESS, trucks)
    - espacios de observación y acción
    - parámetros eléctricos/topológicos de la red

    La red OpenDSS se reconstruirá desde cero en cada step.
    """

    metadata = {"render_modes": ["episode"]}

    # Contrato oficial de observación. Las primeras 102 coordenadas conservan
    # exactamente el contrato de ETAPA 2B; el precio exógeno se añade al final.
    OBS_TIME = slice(0, 2)
    OBS_BUS3_LOAD = slice(2, 8)
    OBS_WH1_LOAD = slice(8, 14)
    OBS_WH2_LOAD = slice(14, 20)
    OBS_PV_AVAILABLE = slice(20, 23)
    OBS_BESS_SOC = slice(23, 24)
    OBS_FORECAST_OFFICE = slice(24, 35)
    OBS_FORECAST_WH1 = slice(35, 46)
    OBS_FORECAST_WH2 = slice(46, 57)
    OBS_FORECAST_PV1 = slice(57, 68)
    OBS_FORECAST_PV2 = slice(68, 79)
    OBS_FORECAST_PV3 = slice(79, 90)
    OBS_TRUCK5 = slice(90, 95)
    OBS_TRUCK6 = slice(95, 100)
    OBS_ZOE_SOC = slice(100, 102)
    OBS_LEGACY_DIM = 102
    OBS_CURRENT_PRICE = slice(102, 103)
    OBS_FORECAST_PRICE = slice(103, 114)
    OBS_DIM = 114

    OBS_LAYOUT = {
        "time": OBS_TIME,
        "bus3_uncontrollable_load": OBS_BUS3_LOAD,
        "warehouse1_load": OBS_WH1_LOAD,
        "warehouse2_load": OBS_WH2_LOAD,
        "pv_available": OBS_PV_AVAILABLE,
        "bess_soc": OBS_BESS_SOC,
        "forecast_office": OBS_FORECAST_OFFICE,
        "forecast_wh1": OBS_FORECAST_WH1,
        "forecast_wh2": OBS_FORECAST_WH2,
        "forecast_pv1": OBS_FORECAST_PV1,
        "forecast_pv2": OBS_FORECAST_PV2,
        "forecast_pv3": OBS_FORECAST_PV3,
        "truck5": OBS_TRUCK5,
        "truck6": OBS_TRUCK6,
        "zoe_soc": OBS_ZOE_SOC,
        "current_price": OBS_CURRENT_PRICE,
        "forecast_price": OBS_FORECAST_PRICE,
    }
    assert OBS_ZOE_SOC.stop == OBS_LEGACY_DIM
    assert OBS_FORECAST_PRICE.stop == OBS_DIM
    assert all(
        current.start == previous.stop
        for previous, current in zip(OBS_LAYOUT.values(), list(OBS_LAYOUT.values())[1:])
    )

    # Contrato externo de acción. Las tres primeras coordenadas activas PV son
    # utilización de potencia, no curtailment: +1 será máximo uso PV y el
    # decoder futuro aplicará curtailment = (1 - action) / 2.
    ACTION_PV1_UTILIZATION = 0
    ACTION_PV1_Q = 1
    ACTION_PV2_UTILIZATION = 2
    ACTION_PV2_Q = 3
    ACTION_PV3_UTILIZATION = 4
    ACTION_PV3_Q = 5
    ACTION_BESS_P = 6
    ACTION_BESS_Q = 7
    ACTION_TRUCK5_P = 8
    ACTION_TRUCK5_Q = 9
    ACTION_TRUCK6_P = 10
    ACTION_TRUCK6_Q = 11
    ACTION_NAMES = (
        "pv1_utilization", "pv1_q", "pv2_utilization", "pv2_q",
        "pv3_utilization", "pv3_q", "bess_p", "bess_q",
        "truck5_p", "truck5_q", "truck6_p", "truck6_q",
    )
    ACTION_DIM = len(ACTION_NAMES)
    assert ACTION_TRUCK6_Q == ACTION_DIM - 1

    # La última acción futura será t=167, para el intervalo [167, 168). Al
    # completar la semana: terminated=True, truncated=False.
    TERMINATION_CONTRACT = "weekly_mdp_terminal_at_t_168"
    ZERO_ACTION_SEMANTICS = (
        "PV al 50% de utilización; BESS y trucks con P=Q=0."
    )

    # Parámetros de episodio y escalas físicas de observación.
    OFFICE_PMAX_KW = 18.74
    WH1_PMAX_KW = 65.66
    WH2_PMAX_KW = 76.12
    PV1_PMAX_KW = 74.25
    PV2_PMAX_KW = 75.58
    PV3_PMAX_KW = 7.75
    ZOE_PMAX_KW = 22.0
    # Perfil económico exógeno importado de otro caso; no proviene del paper
    # industrial, no incluye +0.12 EUR/kWh y aún no distingue compra/venta.
    PRICE_PROFILE_24H = PRICE_PROFILE_24H
    PRICE_SCALE = PRICE_SCALE
    PRICE_RAW_TO_USD_PER_KWH = PRICE_RAW_TO_USD_PER_KWH
    PF_MIN = PF_MIN

    PF_OFFICE_RANGE = (0.80, 0.995)
    PF_WAREHOUSE_RANGE = (0.57, 0.88)
    # No había un intervalo previo en producción; se formaliza el usado en
    # las pruebas de ETAPA 2B para obtener desbalance positivo y acotado.
    PHASE_WEIGHT_RANGE = (0.20, 0.40)

    def __init__(
        self,
        seed: int | None = None,
        *,
        lambda_pf: float = 1.0,
        c_unserved: float = 1.0,
        split: str = "train",
        scenario_id: str | None = None,
        extreme_events: str | ExtremeEventSpec | None = None,
        extreme_event_seed: int | None = None,
        render_mode: str | None = None,
    ):
        """Crea el ambiente sin avanzar su dinámica temporal.

        ``lambda_pf`` es un parámetro externo del modelo, con unidades
        conceptuales USD/(kVAr h), que pondera económicamente las violaciones
        de factor de potencia. No es un hiperparámetro interno de un algoritmo
        RL ni el valor por defecto ``1.0`` procede de un paper o regulación.
        Puede ajustarse por quien configure el ambiente.

        ``c_unserved`` es el costo científico en USD/kWh de déficit energético
        de salida de trucks. La Fase 2 aprobó 1.0 USD/kWh como valor por
        defecto, configurable por el usuario. Debe ser finito y positivo;
        no se calibra aquí ni se incluye entre hiperparámetros de Optuna.
        ``split`` carga únicamente pools del conjunto elegido y permanece fijo.
        ``scenario_id`` fija los cuatro perfiles base; la seed fija su ruido.
        Los estudios iniciales usan ``extreme_events=None``.
        """
        super().__init__()
        self.scenario_catalog = ScenarioCatalog()
        self._split = self.scenario_catalog.validate_split(split)
        self._configured_scenario_id = scenario_id
        if scenario_id is not None:
            self.scenario_catalog.resolve(self.split, scenario_id)
        self.scenario_id: str | None = None
        self._selected_scenario = None
        self._initial_seed_pending = seed is not None
        self._constructor_seed = seed
        # Scientific default explicitly approved for Phase 2; never tuned by Optuna.
        c_unserved = _validate_reward_scalar(c_unserved, name="c_unserved")
        if c_unserved <= 0:
            raise ValueError("c_unserved debe ser positivo (USD/kWh no servido).")
        self.c_unserved = c_unserved
        self._episode_metrics: EpisodeMetrics | None = None

        if render_mode not in (None, "episode"):
            raise ValueError("render_mode debe ser None o 'episode'.")
        self.render_mode = render_mode
        # En el modo de entrenamiento no se crea historial ni se carga
        # Matplotlib. Las referencias de render no participan en la dinámica.
        self._render_history: list[dict[str, Any]] | None = (
            [] if render_mode == "episode" else None
        )
        self._render_episode_metadata: dict[str, Any] | None = None
        self._render_figure = None

        lambda_pf = _validate_reward_scalar(lambda_pf, name="lambda_pf")
        if lambda_pf < 0.0:
            raise ValueError("lambda_pf debe ser mayor o igual que cero.")
        self.lambda_pf = float(lambda_pf)

        # Los eventos se aplican sobre el estado horario efectivo. Los
        # perfiles nominales, sus forecasts y sus RNG siguen aislados.
        self.extreme_events = self._normalize_extreme_events(extreme_events)
        self.extreme_event_seed = self._validate_optional_extreme_event_seed(
            extreme_event_seed
        )
        self.extreme_event_spec: ExtremeEventSpec | None = None
        self.extreme_event_manager: ExtremeEventManager | None = None
        self.extreme_event_schedule_seed: int | None = None

        # =========================================================
        # 1) Semilla
        # =========================================================
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)

        # =========================================================
        # 2) Horizonte temporal
        # =========================================================
        self.horizon = 168
        self.forecast_h = 11
        self.dt_h = 1.0
        self.t = 0
        self.done = False
        self.price_sampler = PriceSampler(
            horizon_hours=self.horizon,
            forecast_h=self.forecast_h,
            seed=10 if seed is None else int(seed) + 10,
        )
        self.price_seed: int | None = None

        # =========================================================
        # 3) Parámetros eléctricos de la red
        # =========================================================
        self.basekv = 0.4   # 400 V LL
        self.grid_basekv = 15.0
        self.phases = 3

        # Fuente 15 kV -> transformador T2 -> anillo LV. Los datos de placa
        # disponibles son 800 kVA, 15/0.4 kV y uk=6%. La hipótesis adoptada
        # para la parte resistiva es R_total=1%; por tanto XHL es la parte
        # reactiva de uk y cada devanado recibe R_total/2 en OpenDSS.
        self.grid_bus = "bus1"
        self.transformer_name = "Tx"
        self.transformer_kva = 800.0
        self.transformer_primary_kv = 15.0
        self.transformer_secondary_kv = self.basekv
        self.transformer_uk_percent = 6.0
        self.transformer_r_percent_total = 1.0
        self.transformer_xhl_percent = math.sqrt(
            self.transformer_uk_percent ** 2 - self.transformer_r_percent_total ** 2
        )
        # benchmark_v1 conserva Wye--Wye sin desplazamiento (Yy0).
        # Aunque Rneut=-1/Xneut=0, los buses sin sufijo conectan el cuarto
        # conductor al nodo 0 en OpenDSS. No hay neutro distribuido explícito.
        self.transformer_primary_connection = "wye"
        self.transformer_secondary_connection = "wye"
        self.transformer_primary_rneut_ohm = -1.0
        self.transformer_primary_xneut_ohm = 0.0
        self.transformer_secondary_rneut_ohm = -1.0
        self.transformer_secondary_xneut_ohm = 0.0
        self.transformer_vector_group_model = "Yy0; neutros referidos al nodo 0; sin conductor neutro distribuido"

        self.line_r_ohm_per_km = 0.08
        self.line_x_ohm_per_km = 0.04

        self.line_lengths_m = {
            "L23": 50.0,
            "L34": 150.0,
            "L45": 20.0,
            "L56": 50.0,
            "L62": 150.0,
        }

        # =========================================================
        # 4) Ubicación de equipos
        # =========================================================
        self.office_bus = "bus3"
        self.bess_bus = "bus4"
        self.wh1_bus = "bus5"
        self.wh2_bus = "bus6"
        self.pv1_bus = "bus5"
        self.pv2_bus = "bus6"
        self.pv3_bus = "bus3"
        self.truck5_bus = "bus5"
        self.truck6_bus = "bus6"

        # Monofásicos
        self.zoe_phase_0 = 1
        self.zoe_phase_1 = 2

        # =========================================================
        # 5) Samplers de carga
        # =========================================================
        self.office_sampler = OfficeLoadSampler(
            filepath=self.scenario_catalog.pool_path(self.split, "office"),
            pmax_target_kw=18.74,
            seed=1 if seed is None else seed + 1,
        )

        self.wh1_sampler = OfficeLoadSampler(
            filepath=self.scenario_catalog.pool_path(self.split, "wh1"),
            pmax_target_kw=65.66,
            seed=2 if seed is None else seed + 2,
        )

        self.wh2_sampler = OfficeLoadSampler(
            filepath=self.scenario_catalog.pool_path(self.split, "wh2"),
            pmax_target_kw=76.12,
            seed=3 if seed is None else seed + 3,
        )

        # =========================================================
        # 6) EVs Renault Zoe (carga exógena monofásica)
        # =========================================================
        self.zoe_fleet = ZoeFleetBus3(
            seed=4 if seed is None else seed + 4
        )

        # =========================================================
        # 7) PV
        # =========================================================
        self.pv1_sampler = WeeklyPVSampler(
            filepath=self.scenario_catalog.pool_path(self.split, "solar"),
            seed=5 if seed is None else seed + 5,
            pmax_target_kw=74.25,
            forecast_sigma_frac=0.10,
            pv_floor=0.5,
        )
        self.pv2_sampler = WeeklyPVSampler(
            filepath=self.scenario_catalog.pool_path(self.split, "solar"),
            seed=6 if seed is None else seed + 6,
            pmax_target_kw=75.58,
            forecast_sigma_frac=0.10,
            pv_floor=0.5,
        )
        self.pv3_sampler = WeeklyPVSampler(####Este es el del edificio de oficinas.
            filepath=self.scenario_catalog.pool_path(self.split, "solar"),
            seed=7 if seed is None else seed + 7,
            pmax_target_kw=7.75,
            forecast_sigma_frac=0.10,
            pv_floor=0.5,
        )

        self.pv1 = PVPlant(
            sampler=self.pv1_sampler,
            s_max_kva=100.0,
            allow_statcom=True,
        )
        self.pv2 = PVPlant(
            sampler=self.pv2_sampler,
            s_max_kva=100,
            allow_statcom=True,
        )
         
        self.pv3 = PVPlant(
            sampler=self.pv3_sampler,
            s_max_kva=10.0,
            allow_statcom=True,
        )

        # =========================================================
        # 8) BESS
        # =========================================================
        self.bess = Battery(BatteryParams())

        # =========================================================
        # 9) Disponibilidad semanal de trucks
        # =========================================================
        self.availability_bus5_7x24 = np.array([
            [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        ], dtype=int)

        self.availability_bus6_7x24 = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        ], dtype=int)

        # =========================================================
        # 10) Trucks V2G trifásicos balanceados
        # =========================================================
        self.truck5 = V2GTruck(
            availability_7x24=self.availability_bus5_7x24,
            seed=8 if seed is None else seed + 8,
        )

        self.truck6 = V2GTruck(
            availability_7x24=self.availability_bus6_7x24,
            seed=9 if seed is None else seed + 9,
        )

        # Todas las acciones externas permanecen en [-1, 1]. El decoder aún
        # no se implementa en esta etapa.
        self.n_actions = self.ACTION_DIM
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_actions,),
            dtype=np.float32,
        )

        # =========================================================
        # 12) Espacio de observación: contrato declarado por OBS_LAYOUT.
        # =========================================================
        self.n_obs = self.OBS_DIM
        self.observation_space = self._build_observation_space()

        # =========================================================
        # 13) Pesos por fase de las cargas (se muestrean en reset)
        # =========================================================
        self.w_office = np.array([1/3, 1/3, 1/3], dtype=float)
        self.w_wh1 = np.array([1/3, 1/3, 1/3], dtype=float)
        self.w_wh2 = np.array([1/3, 1/3, 1/3], dtype=float)
        # Se muestrearán una vez por episodio en el futuro reset().
        self.pf_office = None
        self.pf_wh1 = None
        self.pf_wh2 = None

        # =========================================================
        # 14) Placeholders de cargas de edificios por fase
        # =========================================================
        # Bus 3: Office más Zoe fase 1/fase 2 cuando se implemente el ciclo.
        self.P_office_abc = np.zeros(3, dtype=float)
        self.Q_office_abc = np.zeros(3, dtype=float)
        self.fc_office = np.zeros(self.forecast_h, dtype=float)

        self.P_wh1_abc = np.zeros(3, dtype=float)
        self.Q_wh1_abc = np.zeros(3, dtype=float)
        self.fc_wh1 = np.zeros(self.forecast_h, dtype=float)

        self.P_wh2_abc = np.zeros(3, dtype=float)
        self.Q_wh2_abc = np.zeros(3, dtype=float)
        self.fc_wh2 = np.zeros(self.forecast_h, dtype=float)

        # =========================================================
        # 15) Placeholders de PV
        # =========================================================
        self.P_pv1_avail_now = 0.0
        self.fc_pv1 = np.zeros(self.forecast_h, dtype=float)

        self.P_pv2_avail_now = 0.0
        self.fc_pv2 = np.zeros(self.forecast_h, dtype=float)

        self.P_pv3_avail_now = 0.0
        self.fc_pv3 = np.zeros(self.forecast_h, dtype=float)

        # Precio exógeno: se entrega en observación, nunca en diagnósticos
        # eléctricos. El forecast se prepara por episodio en PriceSampler.
        self.current_price = 0.0
        self.fc_price = np.zeros(self.forecast_h, dtype=float)

        # =========================================================
        # 16) Placeholders Zoe
        # =========================================================
        self.P_zoe_0_now = 0.0
        self.P_zoe_1_now = 0.0

        # =========================================================
        # 17) Últimas acciones aplicadas (útil para logging/debug)
        # =========================================================
        self.last_action = np.zeros(self.n_actions, dtype=float)
        self._episode_initialized = False
        self._episode_terminated = False
        self._hour_state_cache = None
        self.component_seeds = {}
        self.pv_week_idx = None
        self.pv_weather_seed: int | None = None

    def _build_observation_space(self) -> spaces.Box:
        """Construye cotas compatibles con la observación normalizada.

        Los forecasts no se recortan: se permiten valores finitos no acotados
        porque la perturbación gaussiana puede superar ligeramente el pico
        nominal (o ser negativa si se desactiva el clipping del sampler).
        """
        low = np.zeros(self.OBS_DIM, dtype=np.float32)
        high = np.ones(self.OBS_DIM, dtype=np.float32)

        low[self.OBS_TIME] = -1.0
        high[self.OBS_TIME] = 1.0
        # Las realizaciones de demanda se normalizan con Pmax histórico, pero
        # no se recortan a ese pico; las componentes actuales pueden superar
        # uno tras normalizar y deben seguir siendo válidas en el Box.
        for load_slice in (self.OBS_BUS3_LOAD, self.OBS_WH1_LOAD, self.OBS_WH2_LOAD):
            low[load_slice] = 0.0
            high[load_slice] = np.inf
        for forecast_slice in (
            self.OBS_FORECAST_OFFICE,
            self.OBS_FORECAST_WH1,
            self.OBS_FORECAST_WH2,
            self.OBS_FORECAST_PV1,
            self.OBS_FORECAST_PV2,
            self.OBS_FORECAST_PV3,
            self.OBS_FORECAST_PRICE,
        ):
            low[forecast_slice] = -np.inf
            high[forecast_slice] = np.inf
        # PriceSampler conserva realizaciones gaussianas sin recorte. Igual
        # que su forecast, el precio actual normalizado puede superar el
        # máximo nominal o, excepcionalmente, ser negativo.
        low[self.OBS_CURRENT_PRICE] = -np.inf
        high[self.OBS_CURRENT_PRICE] = np.inf
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _validate_current_time(self) -> int:
        if not 0 <= self.t < self.horizon:
            raise RuntimeError(
                f"No hay observación de decisión válida para t={self.t}; "
                f"el horizonte es [0, {self.horizon - 1}]."
            )
        return int(self.t)

    @staticmethod
    def _derive_price_seed(seed: int | None) -> int | None:
        """Deriva una semilla del precio sin consumir el RNG del ambiente."""
        if seed is None:
            return None
        try:
            seed = operator.index(seed)
        except TypeError as exc:
            raise TypeError("seed debe ser un entero o None.") from exc
        return int(np.random.SeedSequence([int(seed), 10]).generate_state(1, dtype=np.uint32)[0])

    @staticmethod
    def _derive_pv_weather_seed(seed: int | None) -> int | None:
        """Deriva la semilla de la meteorología PV común sin tocar otros RNG."""
        if seed is None:
            return None
        try:
            seed = operator.index(seed)
        except TypeError as exc:
            raise TypeError("seed debe ser un entero o None.") from exc
        return int(np.random.SeedSequence([int(seed), 11]).generate_state(1, dtype=np.uint32)[0])

    @staticmethod
    def _validate_optional_extreme_event_seed(seed: int | None) -> int | None:
        """Normaliza una semilla explícita de eventos sin tocar RNG nominal."""
        if seed is None:
            return None
        if isinstance(seed, (bool, np.bool_)):
            raise TypeError("extreme_event_seed debe ser un entero o None, no bool.")
        try:
            return int(operator.index(seed))
        except TypeError as exc:
            raise TypeError("extreme_event_seed debe ser un entero o None.") from exc

    @staticmethod
    def _derive_extreme_event_seed(seed: int | None) -> int | None:
        """Deriva el stream de eventos en un dominio distinto de precio/PV."""
        if seed is None:
            return None
        if isinstance(seed, (bool, np.bool_)):
            raise TypeError("seed debe ser un entero o None, no bool.")
        try:
            seed = operator.index(seed)
        except TypeError as exc:
            raise TypeError("seed debe ser un entero o None.") from exc
        return int(np.random.SeedSequence([int(seed), 12]).generate_state(1, dtype=np.uint32)[0])

    @staticmethod
    def _normalize_extreme_events(
        extreme_events: str | ExtremeEventSpec | None,
    ) -> str | ExtremeEventSpec | None:
        """Normaliza los eventos concretos y el modo de selección aleatoria."""
        if extreme_events is None:
            return None
        if isinstance(extreme_events, ExtremeEventSpec):
            if extreme_events.event_type not in {
                "load_surge",
                "phase_imbalance",
                "pv_collapse",
                "compound",
            }:
                raise NotImplementedError(
                    "MicrorredIndustrial no integra extreme_events='random' por ahora."
                )
            return extreme_events
        if not isinstance(extreme_events, str):
            raise TypeError(
                "extreme_events debe ser None, un string soportado o ExtremeEventSpec."
            )
        normalized = extreme_events.lower()
        if normalized == "none":
            return None
        if normalized in {
            "load_surge",
            "phase_imbalance",
            "pv_collapse",
            "compound",
            "random",
        }:
            return normalized
        raise ValueError(
            "extreme_events debe ser None, 'none', 'load_surge', 'phase_imbalance', "
            "'pv_collapse', 'compound' o 'random'."
        )

    def _prepare_extreme_event_schedule(
        self,
        reset_seed: int | None,
        *,
        solar_base: np.ndarray | None = None,
    ) -> None:
        """Fija el único spec inmutable del episodio sin consumir RNG nominal."""
        self.extreme_event_spec = None
        self.extreme_event_manager = None
        self.extreme_event_schedule_seed = None
        if self.extreme_events is None:
            return

        if isinstance(self.extreme_events, ExtremeEventSpec):
            self.extreme_event_spec = self.extreme_events
        else:
            schedule_seed = self.extreme_event_seed
            if schedule_seed is None:
                schedule_seed = self._derive_extreme_event_seed(reset_seed)
            if schedule_seed is None:
                # reset(seed=None) no promete reproducibilidad, pero esta
                # entropía sigue sin leer self.np_random ni otros componentes.
                schedule_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
            self.extreme_event_schedule_seed = int(schedule_seed)
            event_rng = np.random.default_rng(self.extreme_event_schedule_seed)
            self.extreme_event_spec = generate_extreme_event_spec(
                self.extreme_events,
                event_rng,
                horizon=self.horizon,
                # Para los eventos que alteran irradiancia se reutiliza la
                # misma serie solar base del episodio; no se crea otra
                # meteorología ni se consulta el forecast.
                solar_base=solar_base,
            )

        self.extreme_event_manager = ExtremeEventManager(self.extreme_event_spec)

    def _extreme_event_metadata_at(self, t: int) -> dict[str, Any]:
        """Da metadata estable de evento para reset/step, fuera de la observación."""
        if self.extreme_event_manager is None or self.extreme_event_spec is None:
            return {
                "extreme_event_enabled": False,
                "extreme_event_active": False,
                "extreme_event_type": None,
                "extreme_event_start": None,
                "extreme_event_end": None,
                "extreme_event_target": None,
                "extreme_event_intensity": None,
                "extreme_event_phase": None,
            }

        metadata = self.extreme_event_manager.metadata_at(t)
        load_targets = tuple(self.extreme_event_spec.load_targets)
        pv_targets = tuple(self.extreme_event_spec.pv_targets)
        target: str | list[str] | dict[str, list[str]] | None
        if load_targets and pv_targets:
            target = {
                "load_targets": list(load_targets),
                "pv_targets": list(pv_targets),
            }
        else:
            targets = load_targets or pv_targets
            if len(targets) == 1:
                target = targets[0]
            elif targets:
                target = list(targets)
            else:
                target = None
        return {
            "extreme_event_enabled": True,
            "extreme_event_active": bool(metadata["extreme_event_active"]),
            "extreme_event_type": self.extreme_event_spec.event_type,
            "extreme_event_start": int(self.extreme_event_spec.start_t),
            "extreme_event_end": int(self.extreme_event_spec.end_t),
            "extreme_event_target": target,
            "extreme_event_intensity": dict(metadata["extreme_event_intensity"]),
            "extreme_event_phase": self.extreme_event_spec.dominant_phase,
        }

    def _require_episode_initialized(self) -> None:
        if not self._episode_initialized:
            raise RuntimeError("Llama reset() antes de solicitar una observación.")

    def _collect_hour_state(self, t: int) -> dict:
        """Obtiene el estado físico de hora ``t`` sin mutar ningún componente."""
        if self.pf_office is None or self.pf_wh1 is None or self.pf_wh2 is None:
            raise RuntimeError("Los PF del episodio no están inicializados; llama reset().")

        p_office_total = self.office_sampler.value_at(t)
        q_office_total = self.compute_reactive_power(p_office_total, self.pf_office)
        p_office_abc, q_office_abc = self.split_three_phase(
            p_office_total, q_office_total, self.w_office
        )
        if self.extreme_event_manager is not None:
            p_office_abc, q_office_abc = self.extreme_event_manager.adjust_load_abc(
                t, "office", p_office_abc, q_office_abc
            )

        p_wh1_total = self.wh1_sampler.value_at(t)
        q_wh1_total = self.compute_reactive_power(p_wh1_total, self.pf_wh1)
        p_wh1_abc, q_wh1_abc = self.split_three_phase(
            p_wh1_total, q_wh1_total, self.w_wh1
        )
        if self.extreme_event_manager is not None:
            p_wh1_abc, q_wh1_abc = self.extreme_event_manager.adjust_load_abc(
                t, "wh1", p_wh1_abc, q_wh1_abc
            )

        p_wh2_total = self.wh2_sampler.value_at(t)
        q_wh2_total = self.compute_reactive_power(p_wh2_total, self.pf_wh2)
        p_wh2_abc, q_wh2_abc = self.split_three_phase(
            p_wh2_total, q_wh2_total, self.w_wh2
        )
        if self.extreme_event_manager is not None:
            p_wh2_abc, q_wh2_abc = self.extreme_event_manager.adjust_load_abc(
                t, "wh2", p_wh2_abc, q_wh2_abc
            )

        # get_power_request() es una consulta no mutante. La preparación de
        # eventos de frontera se realiza explícitamente en reset/step futuro.
        p_zoe = self.zoe_fleet.get_power_request(t)
        p_bus3_abc = p_office_abc.copy()
        p_bus3_abc[0] += p_zoe[0]
        p_bus3_abc[1] += p_zoe[1]

        pv_available_nominal = {
            "pv1": float(self.pv1.available_at(t)),
            "pv2": float(self.pv2.available_at(t)),
            "pv3": float(self.pv3.available_at(t)),
        }
        if self.extreme_event_manager is None:
            pv_available_effective = pv_available_nominal.copy()
        else:
            pv_available_effective = self.extreme_event_manager.adjust_pv_available(
                t, pv_available_nominal
            )

        forecasts = {
            "office": self.office_sampler.forecast_at(t, horizon=self.forecast_h),
            "wh1": self.wh1_sampler.forecast_at(t, horizon=self.forecast_h),
            "wh2": self.wh2_sampler.forecast_at(t, horizon=self.forecast_h),
            "pv1": self.pv1.forecast_at(t, horizon=self.forecast_h),
            "pv2": self.pv2.forecast_at(t, horizon=self.forecast_h),
            "pv3": self.pv3.forecast_at(t, horizon=self.forecast_h),
            "price": self.price_sampler.forecast_at(t, horizon=self.forecast_h),
        }
        for name, forecast in forecasts.items():
            if np.asarray(forecast).shape != (self.forecast_h,):
                raise RuntimeError(f"Forecast {name} no tiene shape ({self.forecast_h},).")

        truck5_state = self.truck5.state_at(t)
        truck6_state = self.truck6.state_at(t)
        for name, state in (("truck5", truck5_state), ("truck6", truck6_state)):
            if np.asarray(state).shape != (5,) or not np.all(np.isfinite(state)):
                raise RuntimeError(f"Estado {name} inválido para la observación.")

        return {
            "t": int(t),
            # Se conservan separadas para la telemetría de transición: bus3
            # combina office + Zoe, pero un evento de carga nunca modifica Zoe.
            "p_office_abc": p_office_abc,
            "q_office_abc": q_office_abc,
            "p_bus3_abc": p_bus3_abc,
            "q_bus3_abc": q_office_abc,
            "p_wh1_abc": p_wh1_abc,
            "q_wh1_abc": q_wh1_abc,
            "p_wh2_abc": p_wh2_abc,
            "q_wh2_abc": q_wh2_abc,
            "p_zoe": p_zoe,
            # ``pv_available`` se conserva como vector efectivo por
            # compatibilidad con los consumidores existentes. Los mappings
            # hacen explícita la única disponibilidad usada en esta hora.
            "pv_available_nominal": pv_available_nominal,
            "pv_available_effective": pv_available_effective,
            "pv_available": np.array(
                [
                    pv_available_effective["pv1"],
                    pv_available_effective["pv2"],
                    pv_available_effective["pv3"],
                ],
                dtype=float,
            ),
            "current_price": self.price_sampler.value_at(t),
            "forecasts": forecasts,
            "truck5_state": truck5_state,
            "truck6_state": truck6_state,
            "zoe_soc": self.zoe_fleet.soc.astype(float, copy=True),
        }

    @staticmethod
    def _load_telemetry_from_hour_state(hour_state: dict) -> dict[str, dict[str, np.ndarray | float]]:
        """Construye telemetría de las cargas físicas ya usadas en el snapshot.

        Las cargas son positivas al consumir. Este helper no consulta ni muta
        samplers o componentes: sólo copia arrays efectivos de la hora actual.
        """
        def copy_abc(key: str) -> np.ndarray:
            values = np.asarray(hour_state[key], dtype=float)
            if values.shape != (3,):
                raise RuntimeError(f"{key} debe tener shape (3,) para telemetría de cargas.")
            return values.copy()

        p_office_abc = copy_abc("p_office_abc")
        q_office_abc = copy_abc("q_office_abc")
        p_bus3_abc = copy_abc("p_bus3_abc")
        q_bus3_abc = copy_abc("q_bus3_abc")
        p_wh1_abc = copy_abc("p_wh1_abc")
        q_wh1_abc = copy_abc("q_wh1_abc")
        p_wh2_abc = copy_abc("p_wh2_abc")
        q_wh2_abc = copy_abc("q_wh2_abc")

        p_zoe = np.asarray(hour_state["p_zoe"], dtype=float)
        if p_zoe.shape != (2,):
            raise RuntimeError("p_zoe debe tener shape (2,) para telemetría de cargas.")
        p_zoe_abc = np.array((p_zoe[0], p_zoe[1], 0.0), dtype=float)
        q_zoe_abc = np.zeros(3, dtype=float)

        p_total_abc = p_bus3_abc + p_wh1_abc + p_wh2_abc
        q_total_abc = q_bus3_abc + q_wh1_abc + q_wh2_abc

        def load_entry(p_abc: np.ndarray, q_abc: np.ndarray) -> dict[str, np.ndarray | float]:
            return {
                "p_abc_kw": np.asarray(p_abc, dtype=float).copy(),
                "q_abc_kvar": np.asarray(q_abc, dtype=float).copy(),
                "p_total_kw": float(np.sum(p_abc)),
                "q_total_kvar": float(np.sum(q_abc)),
            }

        return {
            "office": load_entry(p_office_abc, q_office_abc),
            "zoe": load_entry(p_zoe_abc, q_zoe_abc),
            "bus3": load_entry(p_bus3_abc, q_bus3_abc),
            "wh1": load_entry(p_wh1_abc, q_wh1_abc),
            "wh2": load_entry(p_wh2_abc, q_wh2_abc),
            "total": load_entry(p_total_abc, q_total_abc),
        }

    def _cache_hour_state(self, hour_state: dict) -> None:
        """Actualiza placeholders de diagnóstico fuera de la ruta de consulta."""
        self.P_office_abc = hour_state["p_bus3_abc"].copy()
        self.Q_office_abc = hour_state["q_bus3_abc"].copy()
        self.P_wh1_abc = hour_state["p_wh1_abc"].copy()
        self.Q_wh1_abc = hour_state["q_wh1_abc"].copy()
        self.P_wh2_abc = hour_state["p_wh2_abc"].copy()
        self.Q_wh2_abc = hour_state["q_wh2_abc"].copy()
        self.P_zoe_0_now, self.P_zoe_1_now = map(float, hour_state["p_zoe"])
        self.P_pv1_avail_now, self.P_pv2_avail_now, self.P_pv3_avail_now = map(
            float, hour_state["pv_available"]
        )
        self.fc_office = hour_state["forecasts"]["office"].copy()
        self.fc_wh1 = hour_state["forecasts"]["wh1"].copy()
        self.fc_wh2 = hour_state["forecasts"]["wh2"].copy()
        self.fc_pv1 = hour_state["forecasts"]["pv1"].copy()
        self.fc_pv2 = hour_state["forecasts"]["pv2"].copy()
        self.fc_pv3 = hour_state["forecasts"]["pv3"].copy()
        self.current_price = float(hour_state["current_price"])
        self.fc_price = hour_state["forecasts"]["price"].copy()
        self._hour_state_cache = hour_state

    def _prepare_state_at(self, t: int) -> None:
        """Prepara eventos de frontera de ``t`` sin avanzar el intervalo."""
        if t != self.t:
            raise ValueError("El estado a preparar debe coincidir con MicrorredIndustrial.t.")
        self.zoe_fleet.prepare_interval(t)
        self._cache_hour_state(self._collect_hour_state(t))

    def _get_raw_obs(self) -> np.ndarray:
        """Devuelve las 114 variables físicas de ``obs_t`` antes de normalizar."""
        self._require_episode_initialized()
        t = self._validate_current_time()
        hour_state = self._hour_state_cache
        if hour_state is None or hour_state.get("t") != t:
            hour_state = self._collect_hour_state(t)

        raw = np.empty(self.OBS_DIM, dtype=np.float64)
        angle = 2.0 * np.pi * t / self.horizon
        raw[self.OBS_TIME] = (np.sin(angle), np.cos(angle))
        raw[self.OBS_BUS3_LOAD] = np.concatenate(
            (hour_state["p_bus3_abc"], hour_state["q_bus3_abc"])
        )
        raw[self.OBS_WH1_LOAD] = np.concatenate(
            (hour_state["p_wh1_abc"], hour_state["q_wh1_abc"])
        )
        raw[self.OBS_WH2_LOAD] = np.concatenate(
            (hour_state["p_wh2_abc"], hour_state["q_wh2_abc"])
        )
        raw[self.OBS_PV_AVAILABLE] = hour_state["pv_available"]
        raw[self.OBS_BESS_SOC] = self.bess.soc
        raw[self.OBS_FORECAST_OFFICE] = hour_state["forecasts"]["office"]
        raw[self.OBS_FORECAST_WH1] = hour_state["forecasts"]["wh1"]
        raw[self.OBS_FORECAST_WH2] = hour_state["forecasts"]["wh2"]
        raw[self.OBS_FORECAST_PV1] = hour_state["forecasts"]["pv1"]
        raw[self.OBS_FORECAST_PV2] = hour_state["forecasts"]["pv2"]
        raw[self.OBS_FORECAST_PV3] = hour_state["forecasts"]["pv3"]
        raw[self.OBS_TRUCK5] = hour_state["truck5_state"]
        raw[self.OBS_TRUCK6] = hour_state["truck6_state"]
        raw[self.OBS_ZOE_SOC] = hour_state["zoe_soc"]
        raw[self.OBS_CURRENT_PRICE] = hour_state["current_price"]
        raw[self.OBS_FORECAST_PRICE] = hour_state["forecasts"]["price"]

        if raw.shape != (self.OBS_DIM,) or not np.all(np.isfinite(raw)):
            raise RuntimeError(f"La observación física debe tener {self.OBS_DIM} valores finitos.")
        return raw

    def _normalize_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Normaliza con escalas físicas conocidas, sin estadísticas de datos."""
        raw = np.asarray(raw_obs, dtype=np.float64)
        if raw.shape != (self.OBS_DIM,):
            raise ValueError(f"raw_obs debe tener shape ({self.OBS_DIM},).")
        if not np.all(np.isfinite(raw)):
            raise ValueError("raw_obs debe contener únicamente valores finitos.")

        obs = raw.copy()
        bus3_p_scale = np.array(
            [
                self.OFFICE_PMAX_KW + self.ZOE_PMAX_KW,
                self.OFFICE_PMAX_KW + self.ZOE_PMAX_KW,
                self.OFFICE_PMAX_KW,
            ],
            dtype=float,
        )
        office_q_scale = self.OFFICE_PMAX_KW * np.tan(np.arccos(self.PF_OFFICE_RANGE[0]))
        wh_q_scale = self.WH1_PMAX_KW * np.tan(np.arccos(self.PF_WAREHOUSE_RANGE[0]))
        wh2_q_scale = self.WH2_PMAX_KW * np.tan(np.arccos(self.PF_WAREHOUSE_RANGE[0]))

        obs[self.OBS_BUS3_LOAD.start : self.OBS_BUS3_LOAD.start + 3] /= bus3_p_scale
        obs[self.OBS_BUS3_LOAD.start + 3 : self.OBS_BUS3_LOAD.stop] /= office_q_scale
        obs[self.OBS_WH1_LOAD.start : self.OBS_WH1_LOAD.start + 3] /= self.WH1_PMAX_KW
        obs[self.OBS_WH1_LOAD.start + 3 : self.OBS_WH1_LOAD.stop] /= wh_q_scale
        obs[self.OBS_WH2_LOAD.start : self.OBS_WH2_LOAD.start + 3] /= self.WH2_PMAX_KW
        obs[self.OBS_WH2_LOAD.start + 3 : self.OBS_WH2_LOAD.stop] /= wh2_q_scale
        obs[self.OBS_PV_AVAILABLE] /= np.array(
            [self.PV1_PMAX_KW, self.PV2_PMAX_KW, self.PV3_PMAX_KW], dtype=float
        )
        obs[self.OBS_FORECAST_OFFICE] /= self.OFFICE_PMAX_KW
        obs[self.OBS_FORECAST_WH1] /= self.WH1_PMAX_KW
        obs[self.OBS_FORECAST_WH2] /= self.WH2_PMAX_KW
        obs[self.OBS_FORECAST_PV1] /= self.PV1_PMAX_KW
        obs[self.OBS_FORECAST_PV2] /= self.PV2_PMAX_KW
        obs[self.OBS_FORECAST_PV3] /= self.PV3_PMAX_KW
        obs[self.OBS_CURRENT_PRICE] /= self.PRICE_SCALE
        obs[self.OBS_FORECAST_PRICE] /= self.PRICE_SCALE
        return obs

    def _get_obs(self) -> np.ndarray:
        """Observación RL normalizada, finita y compatible con observation_space."""
        obs = self._normalize_obs(self._get_raw_obs()).astype(np.float32)
        if not self.observation_space.contains(obs):
            raise RuntimeError("La observación normalizada está fuera de observation_space.")
        return obs

    def reset(self, *, seed=None, options=None):
        """Inicia un escenario del split con el costo científico configurado.

        ``options={"scenario_id": id}`` selecciona perfiles base sin añadir
        metadata a la observación. Los forecasts siguen siendo realizaciones
        futuras perturbadas de esos mismos perfiles (synthetic noisy forecasts).
        """
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("reset options must be a mapping or None.")
        options = {} if options is None else options
        if set(options) - {"scenario_id"}:
            raise ValueError("reset options only supports scenario_id; split is fixed at construction.")
        requested_id = options.get("scenario_id", self._configured_scenario_id)
        selection = (None if requested_id is None
                     else self.scenario_catalog.resolve(self.split, requested_id))
        if seed is None and self._initial_seed_pending:
            seed = self._constructor_seed
        self._initial_seed_pending = False
        reset_started_at = time.perf_counter()
        super().reset(seed=seed)
        if self._render_history is not None:
            self._render_history.clear()
            self._render_episode_metadata = None
        self.seed_value = seed
        self.rng = self.np_random
        self.t = 0
        self.done = False
        self.last_action.fill(0.0)
        self._episode_initialized = False
        self._episode_terminated = False
        self._hour_state_cache = None

        component_names = ("office", "wh1", "wh2", "pv1", "pv2", "pv3", "truck5", "truck6", "zoe")
        component_values = self.np_random.integers(
            0, np.iinfo(np.uint32).max, size=len(component_names), dtype=np.uint32
        )
        self.component_seeds = {
            name: int(value) for name, value in zip(component_names, component_values)
        }

        # El precio tiene un RNG independiente para no desplazar las muestras
        # existentes de cargas, PV, PF, pesos de fase ni vehículos.
        self.price_seed = self._derive_price_seed(seed)
        self.price_sampler.reset(seed=self.price_seed)

        # Selección independiente de los streams de ruido, PF y reparto ABC.
        if selection is None:
            scenario_rng = np.random.default_rng(
                np.random.SeedSequence([self.component_seeds["office"], 13]))
            selection = self.scenario_catalog.sample(self.split, scenario_rng)
        self._selected_scenario = selection
        self.scenario_id = selection.scenario_id
        self.office_sampler.reset(seed=self.component_seeds["office"],
                                  week_idx=selection.local_indices["office"])
        self.wh1_sampler.reset(seed=self.component_seeds["wh1"],
                               week_idx=selection.local_indices["wh1"])
        self.wh2_sampler.reset(seed=self.component_seeds["wh2"],
                               week_idx=selection.local_indices["wh2"])
        pv_pool_sizes = {
            self.pv1.sampler.weeks_pool.shape[0],
            self.pv2.sampler.weeks_pool.shape[0],
            self.pv3.sampler.weeks_pool.shape[0],
        }
        if len(pv_pool_sizes) != 1:
            raise RuntimeError("Los tres PV deben usar pools solares con igual número de semanas.")
        self.pv_weather_seed = self._derive_pv_weather_seed(seed)
        if self.pv_weather_seed is None:
            self.pv_weather_seed = int(
                np.random.SeedSequence([self.component_seeds["pv1"], 11]).generate_state(
                    1, dtype=np.uint32
                )[0]
            )
        pv_weather_rng = np.random.default_rng(self.pv_weather_seed)
        self.pv_week_idx = selection.local_indices["solar"]
        common_base_weather = self.pv1.sampler.base_weather_from_week(self.pv_week_idx)
        common_real_weather, _ = WeeklyPVSampler.sample_common_real_weather(
            common_base_weather,
            rng=pv_weather_rng,
            realization_sigma_frac=0.10,
        )
        # Las tres plantas comparten una única realización meteorológica. Sus
        # semillas de componente mantienen independientes AR(1) y forecast.
        common_weather_kwargs = {
            "week_idx": self.pv_week_idx,
            "common_base_weather": common_base_weather,
            "common_real_weather": common_real_weather,
        }
        self.pv1.reset(seed=self.component_seeds["pv1"], **common_weather_kwargs)
        self.pv2.reset(seed=self.component_seeds["pv2"], **common_weather_kwargs)
        self.pv3.reset(seed=self.component_seeds["pv3"], **common_weather_kwargs)
        self.bess.reset(soc=0.50)
        self.truck5.reset(seed=self.component_seeds["truck5"])
        self.truck6.reset(seed=self.component_seeds["truck6"])
        self.zoe_fleet.reset(seed=self.component_seeds["zoe"])

        self.pf_office = float(self.sample_power_factor(*self.PF_OFFICE_RANGE, rng=self.np_random))
        self.pf_wh1 = float(self.sample_power_factor(*self.PF_WAREHOUSE_RANGE, rng=self.np_random))
        self.pf_wh2 = float(self.sample_power_factor(*self.PF_WAREHOUSE_RANGE, rng=self.np_random))
        self.w_office = self.sample_phase_weights(*self.PHASE_WEIGHT_RANGE, rng=self.np_random)
        self.w_wh1 = self.sample_phase_weights(*self.PHASE_WEIGHT_RANGE, rng=self.np_random)
        self.w_wh2 = self.sample_phase_weights(*self.PHASE_WEIGHT_RANGE, rng=self.np_random)

        # Debe ocurrir tras toda la aleatoriedad nominal y antes de construir
        # la observación inicial. El stream de eventos no consume self.np_random.
        self._prepare_extreme_event_schedule(seed, solar_base=common_base_weather)

        self._prepare_state_at(self.t)
        self._episode_initialized = True
        obs = self._get_obs()
        info = {
            "t": self.t,
            "pf_office": self.pf_office,
            "pf_wh1": self.pf_wh1,
            "pf_wh2": self.pf_wh2,
            "w_office": self.w_office.copy(),
            "w_wh1": self.w_wh1.copy(),
            "w_wh2": self.w_wh2.copy(),
            "week_indices": {
                "office": self.office_sampler.original_week_idx,
                "wh1": self.wh1_sampler.original_week_idx,
                "wh2": self.wh2_sampler.original_week_idx,
                "pv1": self.pv1.sampler.week_idx,
                "pv2": self.pv2.sampler.week_idx,
                "pv3": self.pv3.sampler.week_idx,
            },
            "pv_week_idx": self.pv_week_idx,
            "current_price": float(self.current_price),
        }
        info.update(self._extreme_event_metadata_at(self.t))
        info.update(self.scenario_catalog.metadata(selection))
        info["c_unserved_usd_per_kwh"] = self.c_unserved
        info["bess_soc_initial"] = float(self.bess.soc)
        info["bess_energy_initial_kwh"] = float(self.bess.E_kWh)
        self._episode_metrics = EpisodeMetrics(
            bess_soc=self.bess.soc, bess_energy_kwh=self.bess.E_kWh,
            started_at=reset_started_at)
        self._capture_render_episode_metadata(info)
        self._episode_metrics.reset_seconds = time.perf_counter() - reset_started_at
        return obs, info

    @property
    def split(self) -> str:
        """Split inmutable de esta instancia; cambiarlo requiere otro ambiente."""
        return self._split

    def get_episode_metrics(self, *, include_runtime: bool = True) -> dict:
        """Copia de las métricas acumuladas; consultarlas no vuelve a agregarlas.

        El tiempo de pared es descriptivo y se excluye con include_runtime=False
        para comparar trayectorias reproducibles. No hay término terminal BESS.
        """
        if self._episode_metrics is None:
            raise RuntimeError("Call reset() before reading episode metrics.")
        result = self._episode_metrics.snapshot(include_runtime=include_runtime)
        result.update({"split": self.split, "scenario_id": self.scenario_id,
                       "c_unserved_usd_per_kwh": self.c_unserved,
                       "lambda_pf": self.lambda_pf})
        return result

    @staticmethod
    def _copy_render_metadata_value(value: Any) -> Any:
        """Copia únicamente contenedores simples de metadata serializable."""
        if isinstance(value, dict):
            return {
                key: MicrorredIndustrial._copy_render_metadata_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [MicrorredIndustrial._copy_render_metadata_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(MicrorredIndustrial._copy_render_metadata_value(item) for item in value)
        return value

    def _capture_render_episode_metadata(self, info: Mapping[str, Any]) -> None:
        """Guarda la metadata fija del episodio sin retener referencias a ``info``."""
        if self._render_history is None:
            return

        enabled = bool(info["extreme_event_enabled"])
        self._render_episode_metadata = {
            "enabled": enabled,
            "event_type": info["extreme_event_type"],
            "start_t": info["extreme_event_start"],
            "end_t": info["extreme_event_end"],
            "target": self._copy_render_metadata_value(info["extreme_event_target"]),
            "intensity": self._copy_render_metadata_value(info["extreme_event_intensity"]),
            "phase": info["extreme_event_phase"],
        }

    def _record_render_step(self, info: Mapping[str, Any]) -> None:
        """Registra una transición completa usando exclusivamente su ``info`` final."""
        if self._render_history is None:
            return

        controls = info["controls"]
        record: dict[str, Any] = {
            "t": int(info["t"]),
            "load_total_kw": float(info["loads"]["total"]["p_total_kw"]),
            "pv1_kw": float(controls["pv1"]["p_inj_kw"]),
            "pv2_kw": float(controls["pv2"]["p_inj_kw"]),
            "pv3_kw": float(controls["pv3"]["p_inj_kw"]),
            "pv1_available_effective_kw": float(
                controls["pv1"]["p_available_effective_kw"]
            ),
            "pv2_available_effective_kw": float(
                controls["pv2"]["p_available_effective_kw"]
            ),
            "pv3_available_effective_kw": float(
                controls["pv3"]["p_available_effective_kw"]
            ),
            "bess_p_kw": float(controls["bess"]["p_inj_kw"]),
            "bess_q_kvar": float(controls["bess"]["q_inj_kvar"]),
            "bess_soc": float(controls["bess"]["soc_after"]),
            "pf_phase": tuple(float(value) for value in info["pf_phase"]),
            "price_raw": float(info["price_raw"]),
            "reward": float(info["reward"]),
        }
        # Estos diagnósticos ya están presentes en ``info`` y quedan listos
        # para extender el dashboard sin repetir consultas a la simulación.
        for key in (
            "vmin_pu",
            "vmax_pu",
            "transformer_loading_pct",
            "max_voltage_unbalance_pct",
            "p_grid_kw",
            "pcc_q_kvar",
        ):
            if key in info:
                record[key] = float(info[key])
        self._render_history.append(record)

    @staticmethod
    def _format_render_event_target(target: Any) -> str:
        """Da una descripción breve y legible del target de un evento."""
        if target is None:
            return ""
        if isinstance(target, dict):
            groups = []
            for name, values in target.items():
                groups.append(f"{name}={','.join(values)}")
            return "; ".join(groups)
        if isinstance(target, (list, tuple)):
            return ",".join(str(value) for value in target)
        return str(target)

    def render(self):
        """Devuelve una figura observacional del episodio registrado hasta ahora."""
        if self.render_mode != "episode":
            raise RuntimeError("render() requiere construir el ambiente con render_mode='episode'.")
        if not self._render_history:
            raise RuntimeError("render() requiere ejecutar al menos un step() antes.")

        # Importación diferida: el modo de entrenamiento nunca carga Matplotlib.
        import matplotlib.pyplot as plt

        if self._render_figure is not None:
            plt.close(self._render_figure)

        history = self._render_history
        hours = [record["t"] for record in history]
        fig, axes = plt.subplots(
            5,
            1,
            sharex=True,
            figsize=(12, 16),
            constrained_layout=True,
        )
        ax_load, ax_pv, ax_bess, ax_pf, ax_price = axes

        ax_load.plot(hours, [record["load_total_kw"] for record in history])
        ax_load.set_title("Total Load")
        ax_load.set_ylabel("Power (kW)")

        pv_lines = []
        for pv_name in ("pv1", "pv2", "pv3"):
            line, = ax_pv.plot(
                hours,
                [record[f"{pv_name}_kw"] for record in history],
                label=f"{pv_name.upper()} injected",
            )
            pv_lines.append((pv_name, line))
        for pv_name, injected_line in pv_lines:
            available_key = f"{pv_name}_available_effective_kw"
            ax_pv.plot(
                hours,
                [record[available_key] for record in history],
                linestyle="--",
                color=injected_line.get_color(),
                alpha=0.70,
                label=f"{pv_name.upper()} available",
            )
        ax_pv.set_title("PV Generation")
        ax_pv.set_ylabel("Power (kW)")
        ax_pv.legend(ncol=2, loc="best")

        bess_line, = ax_bess.plot(
            hours,
            [record["bess_p_kw"] for record in history],
            label="BESS power",
        )
        ax_bess.axhline(0.0, color="0.4", linewidth=0.8)
        ax_bess.set_title("BESS")
        ax_bess.set_ylabel("BESS Power (kW)")
        ax_soc = ax_bess.twinx()
        soc_line, = ax_soc.plot(
            hours,
            [record["bess_soc"] for record in history],
            color="tab:orange",
            label="SOC",
        )
        ax_soc.set_ylabel("SOC")
        ax_soc.set_ylim(0.0, 1.02)
        ax_bess.legend(handles=[bess_line, soc_line], loc="best")

        for phase_index, phase_name in enumerate(("A", "B", "C")):
            ax_pf.plot(
                hours,
                [record["pf_phase"][phase_index] for record in history],
                label=f"PF {phase_name}",
            )
        ax_pf.axhline(self.PF_MIN, color="0.4", linestyle="--", linewidth=0.9, label="PF min")
        ax_pf.set_title("PCC Power Factor")
        ax_pf.set_ylabel("Power Factor")
        ax_pf.set_ylim(0.0, 1.02)
        ax_pf.legend(ncol=4, loc="best")

        ax_price.plot(hours, [record["price_raw"] for record in history])
        ax_price.set_title("Electricity Price")
        ax_price.set_ylabel("¢USD/kWh")
        ax_price.set_xlabel("Hour of week")

        event = self._render_episode_metadata
        if event is not None and event["enabled"]:
            for axis in axes:
                axis.axvspan(event["start_t"], event["end_t"], color="0.35", alpha=0.12)

            title = f"Industrial Microgrid — {event['event_type']} event"
            description = [
                f"hours [{event['start_t']}, {event['end_t']})",
            ]
            target = self._format_render_event_target(event["target"])
            if target:
                description.append(f"target: {target}")
            intensity = event["intensity"] or {}
            if event["event_type"] == "compound":
                description.extend(
                    f"{name.replace('_', ' ')}={value:g}"
                    for name, value in intensity.items()
                )
            elif event["event_type"] == "phase_imbalance":
                description.append(f"dominant phase: {event['phase']}")
                if "phase_share" in intensity:
                    description.append(f"share={intensity['phase_share']:g}")
            else:
                description.extend(
                    f"{name.replace('_', ' ')}={value:g}"
                    for name, value in intensity.items()
                )
            fig.suptitle(f"{title}\n{' · '.join(description)}")
        else:
            fig.suptitle("Industrial Microgrid — Episode Dashboard")

        self._render_figure = fig
        return fig

    def close(self):
        """Cierra únicamente la última figura creada por este ambiente."""
        if self._render_figure is None:
            return
        import matplotlib.pyplot as plt

        plt.close(self._render_figure)
        self._render_figure = None

    def _compute_reward(
        self,
        *,
        p_grid_kw: float,
        price_raw: float,
        pf_phase: np.ndarray,
        q_pcc_kvar_phase: np.ndarray,
        dt_h: float | None = None,
        truck_departure_penalty_usd: float = 0.0,
    ) -> RewardBreakdown:
        """Calcula una reward pura usando el peso configurado del ambiente.

        Solo lee ``lambda_pf`` y, si no se suministra ``dt_h``, el intervalo
        fijo configurado. No consulta perfiles, no resuelve OpenDSS ni muta
        tiempo, SOC o el estado de ningún activo.
        """
        if dt_h is None:
            dt_h = self.dt_h
        return compute_reward(
            p_grid_kw=p_grid_kw,
            price_raw=price_raw,
            pf_phase=pf_phase,
            q_pcc_kvar_phase=q_pcc_kvar_phase,
            dt_h=dt_h,
            lambda_pf=self.lambda_pf,
            truck_departure_penalty_usd=truck_departure_penalty_usd,
        )

    def _decode_action(
        self,
        action,
        t: int | None = None,
        *,
        pv_available: Mapping[str, float] | None = None,
    ) -> dict:
        """Traduce una acción externa 12D sin mutar estado ni aplicar controles.

        La entrada se limita a ``[-1, 1]`` para tolerar desviaciones numéricas
        ligeras de políticas continuas. Una acción cero implica 50 % de
        utilización PV y P/Q nulos para BESS y trucks.
        """
        self._require_episode_initialized()
        try:
            action_array = np.asarray(action, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("action debe ser un vector numérico de longitud 12.") from exc
        if action_array.shape != (self.ACTION_DIM,):
            raise ValueError(f"action debe tener shape ({self.ACTION_DIM},), no {action_array.shape}.")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action debe contener únicamente valores finitos.")
        clipped = np.clip(action_array, -1.0, 1.0)

        if t is None:
            t = self._validate_current_time()
        else:
            try:
                t = operator.index(t)
            except TypeError as exc:
                raise ValueError("t debe ser un índice entero de hora.") from exc
            if not 0 <= t < self.horizon:
                raise ValueError(f"t={t} está fuera del horizonte [0, {self.horizon - 1}].")

        pv_plants = {"pv1": self.pv1, "pv2": self.pv2, "pv3": self.pv3}
        if pv_available is None:
            # Ruta compatible para llamadas directas antiguas. ``step``
            # siempre inyecta el mapping efectivo del estado horario.
            pv_available_by_name = {
                name: float(plant.available_at(t)) for name, plant in pv_plants.items()
            }
        else:
            if not isinstance(pv_available, Mapping):
                raise TypeError("pv_available debe ser un mapping {'pv1': kW, ...}.")
            expected_names = set(pv_plants)
            if set(pv_available) != expected_names:
                raise ValueError("pv_available debe contener exactamente pv1, pv2 y pv3.")
            pv_available_by_name = {}
            for name in pv_plants:
                try:
                    value = float(pv_available[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"pv_available[{name!r}] debe ser numérico.") from exc
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"pv_available[{name!r}] debe ser finito y no negativo.")
                pv_available_by_name[name] = value

        def decode_pv(name: str, plant: PVPlant, active_idx: int, q_idx: int) -> dict:
            active_action = float(clipped[active_idx])
            q_action = float(clipped[q_idx])
            utilization = (active_action + 1.0) / 2.0
            curtailment = 1.0 - utilization
            p_available = pv_available_by_name[name]
            p_command = utilization * p_available
            q_cap = float(np.sqrt(max(0.0, plant.s_max_kva**2 - p_command**2)))
            return {
                "utilization": utilization,
                "curtailment": curtailment,
                "p_available_kw": float(p_available),
                "p_command_kw": float(p_command),
                "q_cap_kvar": q_cap,
                "q_command_kvar": q_action * q_cap,
            }

        p_bess_requested = float(self.bess.params.P_dis_max_kW * clipped[self.ACTION_BESS_P])
        return {
            "t": int(t),
            "action_clipped": tuple(float(value) for value in clipped),
            "pv1": decode_pv("pv1", self.pv1, self.ACTION_PV1_UTILIZATION, self.ACTION_PV1_Q),
            "pv2": decode_pv("pv2", self.pv2, self.ACTION_PV2_UTILIZATION, self.ACTION_PV2_Q),
            "pv3": decode_pv("pv3", self.pv3, self.ACTION_PV3_UTILIZATION, self.ACTION_PV3_Q),
            "bess": {
                "p_requested_inj_kw": p_bess_requested,
                "u_internal_kw": -p_bess_requested,
                "q_frac": float(clipped[self.ACTION_BESS_Q]),
            },
            "truck5": {
                "aP_external": float(clipped[self.ACTION_TRUCK5_P]),
                "aP_internal": -float(clipped[self.ACTION_TRUCK5_P]),
                "aQ_internal": float(clipped[self.ACTION_TRUCK5_Q]),
            },
            "truck6": {
                "aP_external": float(clipped[self.ACTION_TRUCK6_P]),
                "aP_internal": -float(clipped[self.ACTION_TRUCK6_P]),
                "aQ_internal": float(clipped[self.ACTION_TRUCK6_Q]),
            },
        }

    def step(self, action):
        """Ejecuta transaccionalmente la transición de una hora ``[t, t+1)``.

        Primero resuelve controles sin mutar energía, obtiene el snapshot de
        OpenDSS y calcula la reward de la misma hora. Solo después confirma
        BESS/trucks, avanza Zoe y los relojes de los trucks. El final natural
        del horizonte retorna una observación terminal sentinel de 114 ceros.
        """
        self._require_episode_initialized()
        if self._episode_terminated:
            raise RuntimeError("Episode has terminated; call reset() before step().")

        step_started_at = time.perf_counter()
        t = self._validate_current_time()
        # Garantiza la preparación idempotente de Zoe y un estado horario
        # coherente antes de consultar cualquier potencia para [t, t+1).
        self._prepare_state_at(t)
        hour_state = self._hour_state_cache
        if hour_state is None:
            raise RuntimeError("No se pudo preparar el estado horario del ambiente.")
        decoded = self._decode_action(
            action,
            t=t,
            pv_available=hour_state["pv_available_effective"],
        )

        bess_resolution = self.bess.resolve_action(
            u_kW=decoded["bess"]["u_internal_kw"],
            dt_h=self.dt_h,
            q_frac=decoded["bess"]["q_frac"],
        )
        truck5_resolution = self.truck5.resolve_action_at(
            t,
            decoded["truck5"]["aP_internal"],
            decoded["truck5"]["aQ_internal"],
        )
        truck6_resolution = self.truck6.resolve_action_at(
            t,
            decoded["truck6"]["aP_internal"],
            decoded["truck6"]["aQ_internal"],
        )

        snapshot_inputs = ElectricalSnapshotInputs(
            p_bus3_abc_kw=hour_state["p_bus3_abc"],
            q_bus3_abc_kvar=hour_state["q_bus3_abc"],
            p_wh1_abc_kw=hour_state["p_wh1_abc"],
            q_wh1_abc_kvar=hour_state["q_wh1_abc"],
            p_wh2_abc_kw=hour_state["p_wh2_abc"],
            q_wh2_abc_kvar=hour_state["q_wh2_abc"],
            pv1_p_inj_kw=decoded["pv1"]["p_command_kw"],
            pv1_q_inj_kvar=decoded["pv1"]["q_command_kvar"],
            pv2_p_inj_kw=decoded["pv2"]["p_command_kw"],
            pv2_q_inj_kvar=decoded["pv2"]["q_command_kvar"],
            pv3_p_inj_kw=decoded["pv3"]["p_command_kw"],
            pv3_q_inj_kvar=decoded["pv3"]["q_command_kvar"],
            bess_p_inj_kw=bess_resolution.P_grid_kW,
            bess_q_inj_kvar=bess_resolution.Q_set_kvar,
            # La P interna positiva del truck es consumo; el snapshot usa
            # inyección positiva. Q ya conserva la convención validada.
            truck5_p_inj_kw=-truck5_resolution.P_grid_kw,
            truck5_q_inj_kvar=truck5_resolution.Q_grid_kvar,
            truck6_p_inj_kw=-truck6_resolution.P_grid_kw,
            truck6_q_inj_kvar=truck6_resolution.Q_grid_kvar,
        )
        try:
            snapshot = self._solve_electrical_snapshot(snapshot_inputs)
        except Exception as error:
            self._episode_metrics.record_failure(time.perf_counter() - step_started_at, str(error))
            raise
        if not snapshot.converged:
            error = f"OpenDSS did not converge at environment step t={t}."
            self._episode_metrics.record_failure(time.perf_counter() - step_started_at, error)
            raise RuntimeError(error)
        if (
            snapshot.pcc is None
            or snapshot.pcc_pf_phase is None
            or snapshot.pcc_q_kvar_phase is None
        ):
            raise RuntimeError("El snapshot convergente no contiene datos PCC para la reward.")

        # Anticipación pura de salidas programadas usando el SOC resuelto para
        # fin de [t,t+1). Cerrar el horizonte no crea una salida artificial.
        departures = []
        for name, truck, resolution in (
            ("truck5", self.truck5, truck5_resolution),
            ("truck6", self.truck6, truck6_resolution),
        ):
            if t + 1 < self.horizon and truck.connected and not truck.available_at(t + 1):
                departures.append(truck_departure_event(
                    truck=name, departure_t=t + 1, soc_departure=resolution.soc_next,
                    target_soc=truck.soc_target_dep, capacity_kwh=truck.C,
                    c_unserved=self.c_unserved))
        truck_penalty = sum(event["departure_penalty_usd"] for event in departures)
        reward_result = self._compute_reward(
            p_grid_kw=snapshot.pcc["P_total_kW"],
            price_raw=self.price_sampler.value_at(t),
            pf_phase=snapshot.pcc_pf_phase,
            q_pcc_kvar_phase=snapshot.pcc_q_kvar_phase,
            dt_h=self.dt_h,
            truck_departure_penalty_usd=truck_penalty,
        )

        # El snapshot y la reward ya son válidos: este es el único punto de
        # commit energético y temporal de la transición.
        pv_realized = {
            "pv1": self.pv1.apply_resolved_setpoint_at(
                t,
                decoded["pv1"]["p_command_kw"],
                decoded["pv1"]["q_command_kvar"],
                available_kw=hour_state["pv_available_effective"]["pv1"],
            ),
            "pv2": self.pv2.apply_resolved_setpoint_at(
                t,
                decoded["pv2"]["p_command_kw"],
                decoded["pv2"]["q_command_kvar"],
                available_kw=hour_state["pv_available_effective"]["pv2"],
            ),
            "pv3": self.pv3.apply_resolved_setpoint_at(
                t,
                decoded["pv3"]["p_command_kw"],
                decoded["pv3"]["q_command_kvar"],
                available_kw=hour_state["pv_available_effective"]["pv3"],
            ),
        }
        self.bess.commit_resolution(bess_resolution)
        self.truck5.commit_action_at(t, truck5_resolution)
        self.truck6.commit_action_at(t, truck6_resolution)
        self.zoe_fleet.advance_interval(t, hour_state["p_zoe"])
        gap5, _ = self.truck5.advance_interval(t)
        gap6, _ = self.truck6.advance_interval(t)

        next_t = t + 1
        terminated = next_t >= self.horizon
        truncated = False
        self.t = next_t
        self.done = terminated
        self._episode_terminated = terminated
        self.last_action = np.asarray(decoded["action_clipped"], dtype=float)

        controls = {
            name: {
                "p_inj_kw": float(p_q[0]),
                "q_inj_kvar": float(p_q[1]),
                "p_available_nominal_kw": float(hour_state["pv_available_nominal"][name]),
                "p_available_effective_kw": float(hour_state["pv_available_effective"][name]),
            }
            for name, p_q in pv_realized.items()
        }
        controls.update({
            "bess": {
                "p_inj_kw": float(bess_resolution.P_grid_kW),
                "q_inj_kvar": float(bess_resolution.Q_set_kvar),
                "soc_before": float(bess_resolution.soc_before),
                "soc_after": float(bess_resolution.soc_next),
            },
            "truck5": {
                "p_grid_kw": float(truck5_resolution.P_grid_kw),
                "p_inj_kw": float(-truck5_resolution.P_grid_kw),
                "q_inj_kvar": float(truck5_resolution.Q_grid_kvar),
                "soc_before": float(truck5_resolution.soc_before),
                "soc_after": float(truck5_resolution.soc_next),
            },
            "truck6": {
                "p_grid_kw": float(truck6_resolution.P_grid_kw),
                "p_inj_kw": float(-truck6_resolution.P_grid_kw),
                "q_inj_kvar": float(truck6_resolution.Q_grid_kvar),
                "soc_before": float(truck6_resolution.soc_before),
                "soc_after": float(truck6_resolution.soc_next),
            },
            "zoe": {
                "power_kw": hour_state["p_zoe"].astype(float, copy=True),
                "soc_before": hour_state["zoe_soc"].astype(float, copy=True),
                "soc_after": self.zoe_fleet.soc.astype(float, copy=True),
            },
        })
        info = {
            "t": t,
            "next_t": next_t,
            "terminated": bool(terminated),
            "truncated": False,
            "terminal_observation_is_sentinel": bool(terminated),
            "reward": float(reward_result.reward),
            "energy_cost_usd": float(reward_result.energy_cost_usd),
            "operating_cost_usd": float(reward_result.operating_cost_usd),  # alias histórico
            "pf_penalty_usd": float(reward_result.pf_penalty_usd),
            "truck_departure_penalty_usd": float(reward_result.truck_departure_penalty_usd),
            "truck_departures": departures,
            "truck_departure_soc_gaps": {"truck5": gap5, "truck6": gap6},
            "c_unserved_usd_per_kwh": self.c_unserved,
            "scenario_id": self.scenario_id,
            "split": self.split,
            "price_raw": float(reward_result.price_raw),
            "price_usd_per_kwh": float(reward_result.price_usd_per_kwh),
            "p_grid_kw": float(reward_result.p_grid_kw),
            "pf_phase": reward_result.pf_phase.copy(),
            "pf_violation_phase": reward_result.pf_violation_phase.copy(),
            "q_pcc_kvar_phase": reward_result.q_pcc_kvar_phase.copy(),
            "action_clipped": self.last_action.copy(),
            "controls": controls,
            "loads": self._load_telemetry_from_hour_state(hour_state),
        }
        info.update(self._extreme_event_metadata_at(t))
        info.update(snapshot.electrical_info())
        self._record_render_step(info)

        if terminated:
            terminal_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            if not self.observation_space.contains(terminal_obs):
                raise RuntimeError("La observación terminal sentinel es inválida.")
            self._episode_metrics.update(
                info, dt_h=self.dt_h, bess_energy_kwh=self.bess.E_kWh,
                elapsed_seconds=time.perf_counter() - step_started_at)
            info["episode_metrics"] = self.get_episode_metrics(include_runtime=False)
            info["episode_runtime"] = self.get_episode_metrics()["runtime"]
            return terminal_obs, float(reward_result.reward), True, False, info

        self._prepare_state_at(self.t)
        next_obs = self._get_obs()
        self._episode_metrics.update(
            info, dt_h=self.dt_h, bess_energy_kwh=self.bess.E_kWh,
            elapsed_seconds=time.perf_counter() - step_started_at)
        return next_obs, float(reward_result.reward), False, False, info

    def sample_power_factor(self,pf_min: float, pf_max: float, rng=None):
        """
        Muestra un factor de potencia desde una distribución uniforme.

        Parameters
        ----------
        pf_min : float
            Factor de potencia mínimo.
        pf_max : float
            Factor de potencia máximo.
        rng : numpy.random.Generator, optional
            Generador de números aleatorios. Si no se pasa, usa ``self.rng``.

        Returns
        -------
        float
            Factor de potencia muestreado.
        """

        if rng is None:
            rng = self.rng

        return rng.uniform(pf_min, pf_max)
    def sample_phase_weights(self,w_min: float, w_max: float, rng=None):
        """
        Muestra pesos trifásicos positivos que suman exactamente 1.

        Parameters
        ----------
        w_min : float
            Valor mínimo permitido para cada peso.
        w_max : float
            Valor máximo permitido para cada peso.
        rng : numpy.random.Generator, optional
            Generador aleatorio.

        Returns
        -------
        np.ndarray
            Vector [w_a, w_b, w_c] tal que suma 1.
        """
        if rng is None:
            rng = self.rng

        while True:
            w_a = rng.uniform(w_min, w_max)
            w_b = rng.uniform(w_min, w_max)
            w_c = 1.0 - w_a - w_b

            if w_min <= w_c <= w_max:
                return np.array([w_a, w_b, w_c], dtype=float)
            
    def _add_industrial_ring_lines(self, circuit: DSSCircuit) -> None:
        """Añade el anillo LV oficial bus2--bus3--bus4--bus5--bus6--bus2."""
        rph = [self.line_r_ohm_per_km] * 3
        xph = [self.line_x_ohm_per_km] * 3
        circuit.add_unbalanced_line(
            "L23", "bus2", "bus3",
            length=self.line_lengths_m["L23"] / 1000.0,
            units="km",
            r_per_phase=rph,
            x_per_phase=xph
        )
        circuit.add_unbalanced_line(
            "L34", "bus3", "bus4",
            length=self.line_lengths_m["L34"] / 1000.0,
            units="km",
            r_per_phase=rph,
            x_per_phase=xph
        )
        circuit.add_unbalanced_line(
            "L45", "bus4", "bus5",
            length=self.line_lengths_m["L45"] / 1000.0,
            units="km",
            r_per_phase=rph,
            x_per_phase=xph
        )
        circuit.add_unbalanced_line(
            "L56", "bus5", "bus6",
            length=self.line_lengths_m["L56"] / 1000.0,
            units="km",
            r_per_phase=rph,
            x_per_phase=xph
        )
        circuit.add_unbalanced_line(
            "L62", "bus6", "bus2",
            length=self.line_lengths_m["L62"] / 1000.0,
            units="km",
            r_per_phase=rph,
            x_per_phase=xph
        )

    def build_industrial_ring_400V(self):
        """Anillo LV histórico con fuente ideal en bus2, usado por pruebas aisladas."""
        c = DSSCircuit(name="industrial_ring", basekv=self.basekv, phases=self.phases)
        c.connect_to_grid(bus="bus2", pu=1.0, angle=0.0, vsrc_name="Source")
        self._add_industrial_ring_lines(c)
        return c

    def build_industrial_network_with_transformer(self) -> DSSCircuit:
        """Construye la topología de producción: Vsource 15 kV, Tx y anillo LV.

        benchmark_v1 conserva los comandos eléctricos del caso T2: Wye--Wye
        sin desplazamiento (Yy0), Rneut=-1 y buses cuyo neutro se referencia
        efectivamente al nodo 0. No existe un conductor neutro distribuido.
        No se añaden fuentes secundarias ni se altera el anillo.
        """
        circuit = DSSCircuit(
            name="industrial_network_with_transformer",
            basekv=self.grid_basekv,
            phases=self.phases,
        )
        circuit.connect_to_grid(
            bus=self.grid_bus,
            pu=1.0,
            angle=0.0,
            vsrc_name="Source",
        )
        circuit.add_transformer(
            self.transformer_name,
            primary_bus=self.grid_bus,
            secondary_bus="bus2",
            phases=self.phases,
            kva=self.transformer_kva,
            primary_kv=self.transformer_primary_kv,
            secondary_kv=self.transformer_secondary_kv,
            r_percent_total=self.transformer_r_percent_total,
            xhl_percent=self.transformer_xhl_percent,
            primary_connection=self.transformer_primary_connection,
            secondary_connection=self.transformer_secondary_connection,
            primary_rneut_ohm=self.transformer_primary_rneut_ohm,
            primary_xneut_ohm=self.transformer_primary_xneut_ohm,
            secondary_rneut_ohm=self.transformer_secondary_rneut_ohm,
            secondary_xneut_ohm=self.transformer_secondary_xneut_ohm,
        )
        self._add_industrial_ring_lines(circuit)
        circuit.set_voltage_bases([self.grid_basekv, self.basekv])
        return circuit

    @staticmethod
    def _add_snapshot_three_phase_loads(
        circuit: DSSCircuit,
        *,
        element_prefix: str,
        bus: str,
        p_abc_kw: np.ndarray,
        q_abc_kvar: np.ndarray,
        kv_ll: float | None = None,
    ) -> None:
        """Conecta una carga desequilibrada como tres elementos 1φ constantes-P."""
        for phase_number, phase_name in enumerate(("A", "B", "C"), start=1):
            circuit.add_load(
                f"{element_prefix}_{phase_name}",
                f"{bus}.{phase_number}",
                phases=1,
                kw=float(p_abc_kw[phase_number - 1]),
                kvar=float(q_abc_kvar[phase_number - 1]),
                model=1,
                kv_ll=kv_ll,
            )

    @staticmethod
    def _coerce_snapshot_inputs(
        inputs: ElectricalSnapshotInputs | Mapping[str, Any] | None,
        physical_powers: Mapping[str, Any],
    ) -> ElectricalSnapshotInputs:
        """Acepta el dataclass, un dict o campos planos como argumentos keyword."""
        if inputs is None:
            return ElectricalSnapshotInputs.from_mapping(physical_powers)
        if physical_powers:
            raise TypeError("Usa inputs o argumentos de potencia, pero no ambos.")
        if isinstance(inputs, ElectricalSnapshotInputs):
            return inputs
        if isinstance(inputs, Mapping):
            return ElectricalSnapshotInputs.from_mapping(inputs)
        raise TypeError(
            "inputs debe ser ElectricalSnapshotInputs, un mapping o None; "
            f"se recibió {type(inputs).__name__}."
        )

    def _solve_electrical_snapshot(
        self,
        inputs: ElectricalSnapshotInputs | Mapping[str, Any] | None = None,
        /,
        **physical_powers: Any,
    ) -> ElectricalSnapshotResult:
        """Resuelve una única hora estacionaria sin tocar la dinámica del ambiente.

        La red se construye desde cero mediante :class:`DSSCircuit`, cuyo
        constructor emite ``Clear``. Este método recibe exclusivamente
        potencias ya realizadas: no llama al decoder, ni consulta perfiles,
        ni modifica ``t``, SOC o el estado de ningún activo.
        """
        values = self._coerce_snapshot_inputs(inputs, physical_powers)

        # Cada llamada reconstruye la única Vsource en bus1, el transformador
        # 15/0.4 kV, las cinco líneas del anillo y todos los equipos. No hay
        # estado de un snapshot anterior.
        circuit = self.build_industrial_network_with_transformer()
        self._add_snapshot_three_phase_loads(
            circuit,
            element_prefix="Load_bus3",
            bus=self.office_bus,
            p_abc_kw=values.p_bus3_abc_kw,
            q_abc_kvar=values.q_bus3_abc_kvar,
            kv_ll=self.basekv,
        )
        self._add_snapshot_three_phase_loads(
            circuit,
            element_prefix="Load_wh1",
            bus=self.wh1_bus,
            p_abc_kw=values.p_wh1_abc_kw,
            q_abc_kvar=values.q_wh1_abc_kvar,
            kv_ll=self.basekv,
        )
        self._add_snapshot_three_phase_loads(
            circuit,
            element_prefix="Load_wh2",
            bus=self.wh2_bus,
            p_abc_kw=values.p_wh2_abc_kw,
            q_abc_kvar=values.q_wh2_abc_kvar,
            kv_ll=self.basekv,
        )

        # add_fixed_injection conserva aquí P/Q > 0 como inyección física y
        # recibe la potencia total del equipo 3φ (OpenDSS la reparte en fases).
        asset_specs = (
            ("PV1", self.pv1_bus, values.pv1_p_inj_kw, values.pv1_q_inj_kvar),
            ("PV2", self.pv2_bus, values.pv2_p_inj_kw, values.pv2_q_inj_kvar),
            ("PV3", self.pv3_bus, values.pv3_p_inj_kw, values.pv3_q_inj_kvar),
            ("BESS", self.bess_bus, values.bess_p_inj_kw, values.bess_q_inj_kvar),
            ("Truck5", self.truck5_bus, values.truck5_p_inj_kw, values.truck5_q_inj_kvar),
            ("Truck6", self.truck6_bus, values.truck6_p_inj_kw, values.truck6_q_inj_kvar),
        )
        for name, bus, p_inj_kw, q_inj_kvar in asset_specs:
            circuit.add_fixed_injection(
                name,
                bus,
                phases=3,
                kw=p_inj_kw,
                kvar=q_inj_kvar,
                kv_ll=self.basekv,
            )

        converged = circuit.solve()
        diagnostics: dict[str, Any] = {
            "circuit_name": circuit.name,
            "pcc_bus": circuit.pcc_bus,
            "n_lines": 5,
            "n_transformers": 1,
            "n_single_phase_building_loads": 9,
            "n_three_phase_controlled_assets": len(asset_specs),
            "load_model": {
                "model": 1,
                "vminpu": circuit.load_vminpu,
                "vmaxpu": circuit.load_vmaxpu,
                "vlowpu": "OpenDSS default preserved (not edited)",
            },
            "transformer_model": {
                "name": self.transformer_name,
                "kva": self.transformer_kva,
                "primary_kv_ll": self.transformer_primary_kv,
                "secondary_kv_ll": self.transformer_secondary_kv,
                "uk_percent": self.transformer_uk_percent,
                "r_percent_total_assumption": self.transformer_r_percent_total,
                "xhl_percent_assumption": self.transformer_xhl_percent,
                "vector_group_model": self.transformer_vector_group_model,
                "primary_connection": self.transformer_primary_connection,
                "secondary_connection": self.transformer_secondary_connection,
                "primary_rneut_ohm": self.transformer_primary_rneut_ohm,
                "secondary_rneut_ohm": self.transformer_secondary_rneut_ohm,
            },
        }
        if not converged:
            diagnostics["reason"] = "OpenDSS no convergió; no se reportan tensiones ni PCC."
            return ElectricalSnapshotResult(
                converged=False,
                pcc=None,
                voltages_vpu=None,
                voltage_angles_deg=None,
                voltage_phasors_v=None,
                voltage_unbalance_pct=None,
                max_voltage_unbalance_pct=None,
                worst_unbalance_bus=None,
                max_lv_voltage_unbalance_pct=None,
                pcc_p_kw_phase=None,
                pcc_q_kvar_phase=None,
                pcc_pf_phase=None,
                pcc_pf_total=None,
                vmin_pu=None,
                vmax_pu=None,
                vmin_bus=None,
                vmin_phase=None,
                vmax_bus=None,
                vmax_phase=None,
                p_losses_kw=None,
                q_losses_kvar=None,
                power_balance=None,
                balance_residual_kw=None,
                element_powers=None,
                diagnostics=diagnostics,
                bus2_vpu=None,
                transformer_loading_percent=None,
                transformer_primary_p_kw=None,
                transformer_primary_q_kvar=None,
                transformer_secondary_p_kw=None,
                transformer_secondary_q_kvar=None,
                transformer_p_losses_kw=None,
                transformer_q_losses_kvar=None,
            )

        pcc_df, pcc_debug_df = circuit.substation_phase_powers_fixed(self.grid_bus)
        pcc_by_phase = {
            int(row["phase"]): row
            for _, row in pcc_df[pcc_df["phase"] != "TOTAL"].iterrows()
        }
        if set(pcc_by_phase) != {1, 2, 3}:
            raise RuntimeError(f"El PCC no devolvió exactamente tres fases: {sorted(pcc_by_phase)}.")
        pcc = {
            "P_a_kW": float(pcc_by_phase[1]["P_kW"]),
            "P_b_kW": float(pcc_by_phase[2]["P_kW"]),
            "P_c_kW": float(pcc_by_phase[3]["P_kW"]),
            "Q_a_kvar": float(pcc_by_phase[1]["Q_kvar"]),
            "Q_b_kvar": float(pcc_by_phase[2]["Q_kvar"]),
            "Q_c_kvar": float(pcc_by_phase[3]["Q_kvar"]),
        }
        pcc["P_total_kW"] = pcc["P_a_kW"] + pcc["P_b_kW"] + pcc["P_c_kW"]
        pcc["Q_total_kvar"] = pcc["Q_a_kvar"] + pcc["Q_b_kvar"] + pcc["Q_c_kvar"]
        # Alias compactos solicitados en el contrato de salida.
        pcc.update({
            "P_a": pcc["P_a_kW"], "P_b": pcc["P_b_kW"], "P_c": pcc["P_c_kW"],
            "P_total": pcc["P_total_kW"],
            "Q_a": pcc["Q_a_kvar"], "Q_b": pcc["Q_b_kvar"], "Q_c": pcc["Q_c_kvar"],
            "Q_total": pcc["Q_total_kvar"],
        })
        pcc_p_kw_phase = np.array(
            [pcc["P_a_kW"], pcc["P_b_kW"], pcc["P_c_kW"]], dtype=float
        )
        pcc_q_kvar_phase = np.array(
            [pcc["Q_a_kvar"], pcc["Q_b_kvar"], pcc["Q_c_kvar"]], dtype=float
        )
        pcc_pf_phase = np.array(
            [
                calculate_power_factor(p_kw, q_kvar)
                for p_kw, q_kvar in zip(pcc_p_kw_phase, pcc_q_kvar_phase)
            ],
            dtype=float,
        )
        pcc_pf_total = calculate_power_factor(pcc["P_total_kW"], pcc["Q_total_kvar"])
        pcc.update({
            "PF_a": float(pcc_pf_phase[0]),
            "PF_b": float(pcc_pf_phase[1]),
            "PF_c": float(pcc_pf_phase[2]),
            "PF_total": pcc_pf_total,
        })

        reported_buses = (self.grid_bus, "bus2", "bus3", "bus4", "bus5", "bus6")
        voltage_df = circuit.get_bus_voltages_df(list(reported_buses))
        voltages_vpu: dict[str, np.ndarray] = {}
        voltage_angles_deg: dict[str, np.ndarray] = {}
        voltage_phasors_v: dict[str, np.ndarray] = {}
        voltage_unbalance_pct: dict[str, float] = {}
        for bus in reported_buses:
            bus_rows = voltage_df[voltage_df["bus"].str.lower() == bus]
            phases = {
                int(row["phase"]): (
                    float(row["V_pu"]),
                    float(row["V_mag_V"]),
                    float(row["V_ang_deg"]),
                )
                for _, row in bus_rows.iterrows()
            }
            if set(phases) != {1, 2, 3} or not np.all(
                np.isfinite(np.asarray(list(phases.values()), dtype=float))
            ):
                raise RuntimeError(f"OpenDSS devolvió tensiones inválidas para {bus}: {phases}.")
            ordered_values = [phases[phase] for phase in (1, 2, 3)]
            voltages_vpu[bus] = np.array(
                [values[0] for values in ordered_values], dtype=float
            )
            voltage_angles_deg[bus] = np.array(
                [values[2] for values in ordered_values], dtype=float
            )
            # Se usan magnitudes y ángulos realmente resueltos por OpenDSS;
            # nunca se infiere un desfase ideal de +/-120 grados a partir de Vpu.
            voltage_phasors_v[bus] = np.array(
                [
                    magnitude_v * np.exp(1j * np.deg2rad(angle_deg))
                    for _, magnitude_v, angle_deg in ordered_values
                ],
                dtype=np.complex128,
            )
            voltage_unbalance_pct[bus] = calculate_voltage_unbalance_pct(
                voltage_phasors_v[bus]
            )

        voltage_extrema = [
            (float(value), bus, PHASE_LABELS[phase_index])
            for bus, values_abc in voltages_vpu.items()
            for phase_index, value in enumerate(values_abc)
        ]
        vmin_pu, vmin_bus, vmin_phase = min(voltage_extrema, key=lambda item: item[0])
        vmax_pu, vmax_bus, vmax_phase = max(voltage_extrema, key=lambda item: item[0])
        worst_unbalance_bus = max(voltage_unbalance_pct, key=voltage_unbalance_pct.get)
        max_voltage_unbalance_pct = voltage_unbalance_pct[worst_unbalance_bus]
        lv_unbalance = {
            bus: value for bus, value in voltage_unbalance_pct.items() if bus != self.grid_bus
        }
        max_lv_voltage_unbalance_pct = max(lv_unbalance.values())

        power_balance = circuit.get_power_balance()
        transformer_primary_pairs = circuit._terminal_power_pairs(
            f"Transformer.{self.transformer_name}", terminal_index=0
        )
        transformer_secondary_pairs = circuit._terminal_power_pairs(
            f"Transformer.{self.transformer_name}", terminal_index=1
        )
        transformer_primary_p_kw = float(sum(pair[0] for pair in transformer_primary_pairs))
        transformer_primary_q_kvar = float(sum(pair[1] for pair in transformer_primary_pairs))
        # En el terminal secundario, el signo nativo positivo entra al
        # transformador. Se invierte para reportar entrega positiva hacia LV.
        transformer_secondary_p_kw = float(-sum(pair[0] for pair in transformer_secondary_pairs))
        transformer_secondary_q_kvar = float(-sum(pair[1] for pair in transformer_secondary_pairs))
        transformer_p_losses_kw = transformer_primary_p_kw - transformer_secondary_p_kw
        transformer_q_losses_kvar = transformer_primary_q_kvar - transformer_secondary_q_kvar
        transformer_loading_percent = 100.0 * math.hypot(
            transformer_primary_p_kw, transformer_primary_q_kvar
        ) / self.transformer_kva
        p_load_total_kw = float(
            values.p_bus3_abc_kw.sum() + values.p_wh1_abc_kw.sum() + values.p_wh2_abc_kw.sum()
        )
        q_load_total_kvar = float(
            values.q_bus3_abc_kvar.sum() + values.q_wh1_abc_kvar.sum() + values.q_wh2_abc_kvar.sum()
        )
        p_local_injection_kw = float(sum(spec[2] for spec in asset_specs))
        q_local_injection_kvar = float(sum(spec[3] for spec in asset_specs))
        balance_residual_kw = float(
            pcc["P_total_kW"] + p_local_injection_kw - p_load_total_kw
            - power_balance["P_losses_kW"]
        )
        balance_residual_kvar = float(
            pcc["Q_total_kvar"] + q_local_injection_kvar - q_load_total_kvar
            - power_balance["Q_losses_kvar"]
        )
        power_balance.update({
            "P_load_total_kW": p_load_total_kw,
            "Q_load_total_kvar": q_load_total_kvar,
            "P_local_injection_kW": p_local_injection_kw,
            "Q_local_injection_kvar": q_local_injection_kvar,
            "physical_balance_residual_kW": balance_residual_kw,
            "physical_balance_residual_kvar": balance_residual_kvar,
        })

        element_powers: dict[str, dict[str, np.ndarray]] = {}
        for name, _, _, _ in asset_specs:
            # CktElement.Powers() reporta la potencia que entra al Load
            # negativo. Se invierte para exponer nuevamente inyección física.
            terminal_pairs = circuit._terminal_power_pairs(f"Load.{name}", terminal_index=0)
            if len(terminal_pairs) < 3:
                raise RuntimeError(f"El equipo {name} no devolvió tres fases eléctricas.")
            element_powers[name.lower()] = {
                "P_inj_abc_kW": -np.asarray([pair[0] for pair in terminal_pairs[:3]], dtype=float),
                "Q_inj_abc_kvar": -np.asarray([pair[1] for pair in terminal_pairs[:3]], dtype=float),
            }

        diagnostics["pcc_debug"] = pcc_debug_df.to_dict(orient="records")
        diagnostics["voltage_unbalance"] = {
            "phase_order": list(PHASE_LABELS),
            "definition": "100 * abs(V2) / abs(V1) (negative-sequence / positive-sequence)",
            "max_pct": max_voltage_unbalance_pct,
            "worst_bus": worst_unbalance_bus,
            "max_lv_pct": max_lv_voltage_unbalance_pct,
        }
        diagnostics["opendss_balance_residual_kW"] = float(power_balance["balance_residual_kW"])
        diagnostics["opendss_balance_residual_kvar"] = float(power_balance["balance_residual_kvar"])
        diagnostics["transformer"] = {
            "loading_percent": transformer_loading_percent,
            "primary_p_kw": transformer_primary_p_kw,
            "primary_q_kvar": transformer_primary_q_kvar,
            "secondary_delivery_p_kw": transformer_secondary_p_kw,
            "secondary_delivery_q_kvar": transformer_secondary_q_kvar,
            "p_losses_kw": transformer_p_losses_kw,
            "q_losses_kvar": transformer_q_losses_kvar,
        }
        return ElectricalSnapshotResult(
            converged=True,
            pcc=pcc,
            voltages_vpu=voltages_vpu,
            voltage_angles_deg=voltage_angles_deg,
            voltage_phasors_v=voltage_phasors_v,
            voltage_unbalance_pct=voltage_unbalance_pct,
            max_voltage_unbalance_pct=max_voltage_unbalance_pct,
            worst_unbalance_bus=worst_unbalance_bus,
            max_lv_voltage_unbalance_pct=max_lv_voltage_unbalance_pct,
            pcc_p_kw_phase=pcc_p_kw_phase,
            pcc_q_kvar_phase=pcc_q_kvar_phase,
            pcc_pf_phase=pcc_pf_phase,
            pcc_pf_total=pcc_pf_total,
            vmin_pu=vmin_pu,
            vmax_pu=vmax_pu,
            vmin_bus=vmin_bus,
            vmin_phase=vmin_phase,
            vmax_bus=vmax_bus,
            vmax_phase=vmax_phase,
            p_losses_kw=float(power_balance["P_losses_kW"]),
            q_losses_kvar=float(power_balance["Q_losses_kvar"]),
            power_balance=power_balance,
            balance_residual_kw=balance_residual_kw,
            element_powers=element_powers,
            diagnostics=diagnostics,
            bus2_vpu=voltages_vpu["bus2"].copy(),
            transformer_loading_percent=transformer_loading_percent,
            transformer_primary_p_kw=transformer_primary_p_kw,
            transformer_primary_q_kvar=transformer_primary_q_kvar,
            transformer_secondary_p_kw=transformer_secondary_p_kw,
            transformer_secondary_q_kvar=transformer_secondary_q_kvar,
            transformer_p_losses_kw=transformer_p_losses_kw,
            transformer_q_losses_kvar=transformer_q_losses_kvar,
        )

    def compute_reactive_power(self,P: float, pf: float):
        """
        Calcula la potencia reactiva Q a partir de P y factor de potencia pf.

        Parameters
        ----------
        P : float
            Potencia activa.
        pf : float
            Factor de potencia, debe estar entre 0 y 1.

        Returns
        -------
        float
            Potencia reactiva Q.
        """
        pf = np.clip(pf, 1e-6, 1.0)
        return P * np.tan(np.arccos(pf))
    
    def split_three_phase(self,P_total: float, Q_total: float, weights):
        """
        Reparte P y Q total en tres fases según un vector de pesos.

        Parameters
        ----------
        P_total : float
            Potencia activa total.
        Q_total : float
            Potencia reactiva total.
        weights : array-like
            Vector [w_a, w_b, w_c] que suma 1.

        Returns
        -------
        tuple
            (P_abc, Q_abc), ambos arrays de longitud 3.
        """
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

        P_abc = P_total * weights
        Q_abc = Q_total * weights

        return P_abc, Q_abc
    
            
   
