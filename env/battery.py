from dataclasses import dataclass
from typing import Optional
import math
import numpy as np


@dataclass
class BatteryParams:
    # Defaults alineados con el paper del BESS centralizado en bus 4
    E_nominal_kWh: float = 250.0
    E_max_kWh: float = 0.98 * 250.0   # 245 kWh
    E_min_kWh: float = 0.08 * 250.0   # 20 kWh
    P_ch_max_kW: float = 180.0
    P_dis_max_kW: float = 180.0
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    rho_per_h: float = 0.0
    S_rated_kVA: Optional[float] = 180.0
    Qmax_kvar: Optional[float] = 180.0


@dataclass(frozen=True)
class BatteryActionResolution:
    """Resultado inmutable de resolver una acción del BESS sin mutarlo."""

    u_requested_kW: float
    q_frac_requested: float
    q_frac_applied: float
    dt_h: float
    p_plus_kW: float
    p_minus_kW: float
    P_grid_kW: float
    Q_set_kvar: float
    S_used_kVA: float
    pf_out: float
    E_before_kWh: float
    E_next_kWh: float
    soc_before: float
    soc_next: float


class Battery:
    """
    Modelo de batería con dinámica de energía y control de potencia activa/reactiva.

    Convención:
      - u_kW > 0  => cargar (absorber de la red)
      - u_kW < 0  => descargar (inyectar a la red)

    Q:
      - q_frac in [-1, 1]
      - permite inyectar y absorber reactiva
      - solo está limitada por:
            1) Qmax_kvar
            2) la envolvente del inversor: P^2 + Q^2 <= S_rated^2
    """

    def __init__(self, params: BatteryParams = BatteryParams(), E_init_kWh: Optional[float] = None):
        self.params = params

        if self.params.S_rated_kVA is None:
            self.S_rated_kVA = max(self.params.P_ch_max_kW, self.params.P_dis_max_kW)
        else:
            self.S_rated_kVA = float(self.params.S_rated_kVA)

        if self.params.Qmax_kvar is None:
            self.Qmax_kvar = self.S_rated_kVA
        else:
            self.Qmax_kvar = float(self.params.Qmax_kvar)

        # Estado inicial físico de referencia: 50% de la capacidad nominal.
        if E_init_kWh is None:
            self.E_kWh = 0.50 * self.params.E_nominal_kWh
        else:
            self.E_kWh = float(E_init_kWh)

        self.E_kWh = min(max(self.E_kWh, self.params.E_min_kWh), self.params.E_max_kWh)

        self.p_plus_kW = 0.0
        self.p_minus_kW = 0.0
        self.Q_set_kvar = 0.0
        self.last_P_grid_kW = 0.0
        self.last_S_used_kVA = 0.0
        self.last_pf_out = 1.0

    @property
    def soc(self) -> float:
        """SOC físico respecto a la capacidad nominal de la batería."""
        denom = self.params.E_nominal_kWh
        if denom <= 0:
            return 0.0
        return self.E_kWh / denom

    def reset(
        self,
        E_init_kWh: Optional[float] = None,
        soc: Optional[float] = None,
        clear_history: bool = True,
    ):
        if soc is not None:
            self.E_kWh = float(soc) * self.params.E_nominal_kWh
        elif E_init_kWh is not None:
            self.E_kWh = float(E_init_kWh)
        else:
            self.E_kWh = 0.50 * self.params.E_nominal_kWh

        self.E_kWh = min(max(self.E_kWh, self.params.E_min_kWh), self.params.E_max_kWh)

        if clear_history:
            self.p_plus_kW = 0.0
            self.p_minus_kW = 0.0
            self.Q_set_kvar = 0.0
            self.last_P_grid_kW = 0.0
            self.last_S_used_kVA = 0.0
            self.last_pf_out = 1.0

    def resolve_action(
        self,
        u_kW: float,
        dt_h: float = 1.0,
        q_frac: float = 0.0,
    ) -> BatteryActionResolution:
        """Resuelve una acción sin cambiar energía, historial ni estados del BESS."""
        if u_kW >= 0.0:
            p_plus = min(u_kW, self.params.P_ch_max_kW)
            p_minus = 0.0
        else:
            p_plus = 0.0
            p_minus = min(-u_kW, self.params.P_dis_max_kW)

        E = self.E_kWh
        rho = self.params.rho_per_h

        if p_plus > 0.0:
            energy_room_kWh = max(self.params.E_max_kWh - E, 0.0)
            p_plus_cap = (energy_room_kWh / dt_h + rho * E) / max(self.params.eta_ch, 1e-9)
            p_plus = min(p_plus, p_plus_cap)

        if p_minus > 0.0:
            energy_available_kWh = max(E - self.params.E_min_kWh, 0.0)
            p_minus_cap = max(
                (energy_available_kWh / dt_h - rho * E) * max(self.params.eta_dis, 1e-9),
                0.0,
            )
            p_minus = min(p_minus, p_minus_cap)

        P_grid = p_minus - p_plus

        if self.S_rated_kVA is not None and abs(P_grid) > self.S_rated_kVA:
            if P_grid > 0:
                p_minus = self.S_rated_kVA
            else:
                p_plus = self.S_rated_kVA
            P_grid = p_minus - p_plus

        q_frac_applied = float(np.clip(q_frac, -1.0, 1.0))
        Q_req = q_frac_applied * self.Qmax_kvar

        if self.S_rated_kVA is not None:
            rem = max(self.S_rated_kVA**2 - P_grid**2, 0.0)
            Q_S_cap = math.sqrt(rem) if rem > 0.0 else 0.0
        else:
            Q_S_cap = self.Qmax_kvar

        Q_cap = min(self.Qmax_kvar, Q_S_cap)
        Q_set = float(np.clip(Q_req, -Q_cap, Q_cap))

        deltaE = dt_h * (
            self.params.eta_ch * p_plus
            - (p_minus / max(self.params.eta_dis, 1e-9))
            - rho * E
        )
        E_next = E + deltaE
        E_next = min(max(E_next, self.params.E_min_kWh), self.params.E_max_kWh)

        S_used = math.sqrt(P_grid**2 + Q_set**2)
        pf_out = float(abs(P_grid) / S_used) if S_used > 1e-12 else 1.0

        nominal_energy = self.params.E_nominal_kWh
        soc_before = float(E / nominal_energy) if nominal_energy > 0 else 0.0
        soc_next = float(E_next / nominal_energy) if nominal_energy > 0 else 0.0

        return BatteryActionResolution(
            u_requested_kW=float(u_kW),
            q_frac_requested=float(q_frac),
            q_frac_applied=q_frac_applied,
            dt_h=float(dt_h),
            p_plus_kW=float(p_plus),
            p_minus_kW=float(p_minus),
            P_grid_kW=float(P_grid),
            Q_set_kvar=float(Q_set),
            S_used_kVA=float(S_used),
            pf_out=pf_out,
            E_before_kWh=float(E),
            E_next_kWh=float(E_next),
            soc_before=soc_before,
            soc_next=soc_next,
        )

    def commit_resolution(self, resolution: BatteryActionResolution):
        """Confirma una resolución vigente; rechaza previews calculados sobre otro SOC."""
        if not isinstance(resolution, BatteryActionResolution):
            raise TypeError("resolution debe ser un BatteryActionResolution.")
        if not math.isclose(
            resolution.E_before_kWh,
            self.E_kWh,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("La resolución del BESS está desactualizada.")

        self.E_kWh = resolution.E_next_kWh
        self.p_plus_kW = resolution.p_plus_kW
        self.p_minus_kW = resolution.p_minus_kW
        self.Q_set_kvar = resolution.Q_set_kvar
        self.last_P_grid_kW = resolution.P_grid_kW
        self.last_S_used_kVA = resolution.S_used_kVA
        self.last_pf_out = resolution.pf_out

        return {
            "E_kWh": self.E_kWh,
            "soc": self.soc,
            "p_plus_kW": self.p_plus_kW,
            "p_minus_kW": self.p_minus_kW,
            "P_grid_kW": self.last_P_grid_kW,
            "Q_set_kvar": self.Q_set_kvar,
            "S_used_kVA": self.last_S_used_kVA,
            "pf_out": self.last_pf_out,
        }

    def step(self, u_kW: float, dt_h: float = 1.0, q_frac: float = 0.0):
        """API mutable legacy: resuelve y confirma una acción en un solo llamado."""
        return self.commit_resolution(self.resolve_action(u_kW, dt_h=dt_h, q_frac=q_frac))
