"""Economic departure events and descriptive episode accounting.

Voltage thresholds are metrics only. Runtime is observational and excluded from
trajectory equality; no statistic here changes control, power flow, SOC or reward.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np


def truck_departure_event(
    *, truck: str, departure_t: int, soc_departure: float,
    target_soc: float, capacity_kwh: float, c_unserved: float,
) -> dict:
    values = (soc_departure, target_soc, capacity_kwh, c_unserved)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Departure accounting requires finite physical inputs.")
    if not 0 <= soc_departure <= 1 or not 0 <= target_soc <= 1:
        raise ValueError("Departure SOC and target must belong to [0, 1].")
    if capacity_kwh <= 0 or c_unserved <= 0:
        raise ValueError("capacity_kwh and explicit c_unserved must be positive.")
    deficit = max(0.0, (target_soc - soc_departure) * capacity_kwh)
    return {
        "truck": truck, "departure_t": int(departure_t),
        "soc_departure": float(soc_departure), "target_soc": float(target_soc),
        "energy_deficit_kwh": float(deficit),
        "departure_penalty_usd": float(c_unserved * deficit),
        "departure_violation": bool(deficit > 0.0),
    }


class EpisodeMetrics:
    voltage_limits_pu = (0.95, 1.05)
    pf_min = 0.90

    def __init__(self, *, bess_soc: float, bess_energy_kwh: float, started_at: float):
        self.started_at = started_at
        self.finished_at = None
        self.reset_seconds = 0.0
        self.step_seconds_total = 0.0
        self.step_seconds_max = 0.0
        self.steps = 0
        self.complete = False
        self.failed_solver_attempts = 0
        self.last_solver_error = None
        self.totals = {
            "return_total": 0.0, "energy_cost_usd": 0.0, "pf_penalty_usd": 0.0,
            "truck_departure_penalty_usd": 0.0, "import_kwh": 0.0,
            "export_kwh": 0.0, "losses_kwh": 0.0,
            "pv_available_kwh": 0.0, "pv_utilized_kwh": 0.0,
            "curtailment_kwh": 0.0,
        }
        self.pv_by_plant = {
            name: {"available_kwh": 0.0, "utilized_kwh": 0.0, "curtailment_kwh": 0.0}
            for name in ("pv1", "pv2", "pv3")
        }
        self.voltage_min_phase = np.full(3, np.inf)
        self.voltage_max_phase = np.full(3, -np.inf)
        self.voltage_violation_steps = 0
        self.voltage_violation_steps_phase = np.zeros(3, dtype=int)
        self.voltage_violation_node_phase_count = 0
        self.voltage_violation_max_pu = 0.0
        self.voltage_violation_sum_pu = 0.0
        self.voltage_violation_integral_pu_h = 0.0
        self.pf_min_phase = np.full(3, np.inf)
        self.pf_max_phase = np.full(3, -np.inf)
        self.pf_sum_phase = np.zeros(3)
        self.pf_last_phase = None
        self.pf_violation_steps = 0
        self.pf_violation_steps_phase = np.zeros(3, dtype=int)
        self.pf_violation_max = 0.0
        self.pf_violation_sum = 0.0
        self.bess_soc_initial = float(bess_soc)
        self.bess_soc_final = float(bess_soc)
        self.bess_energy_initial_kwh = float(bess_energy_kwh)
        self.bess_energy_final_kwh = float(bess_energy_kwh)
        self.departures = []
        self.vuf_max_pct = 0.0
        self.transformer_loading_max_pct = 0.0
        self.balance_residual_max_abs_kw = 0.0

    def record_failure(self, elapsed_seconds: float, error: str) -> None:
        self.failed_solver_attempts += 1
        self.last_solver_error = str(error)
        self._record_runtime(elapsed_seconds)

    def _record_runtime(self, elapsed_seconds: float) -> None:
        self.step_seconds_total += elapsed_seconds
        self.step_seconds_max = max(self.step_seconds_max, elapsed_seconds)

    def update(self, info: dict, *, dt_h: float, bess_energy_kwh: float, elapsed_seconds: float):
        self.steps += 1
        self.complete = bool(info["terminated"])
        if self.complete:
            self.finished_at = time.perf_counter()
        self._record_runtime(elapsed_seconds)
        for key in ("energy_cost_usd", "pf_penalty_usd", "truck_departure_penalty_usd"):
            self.totals[key] += info[key]
        self.totals["return_total"] += info["reward"]
        self.totals["import_kwh"] += max(info["p_grid_kw"], 0.0) * dt_h
        self.totals["export_kwh"] += max(-info["p_grid_kw"], 0.0) * dt_h
        self.totals["losses_kwh"] += info["p_losses_kw"] * dt_h
        for name, totals in self.pv_by_plant.items():
            control = info["controls"][name]
            available = control["p_available_effective_kw"] * dt_h
            utilized = control["p_inj_kw"] * dt_h
            curtailed = max(0.0, available - utilized)
            for key, value in zip(("available_kwh", "utilized_kwh", "curtailment_kwh"),
                                  (available, utilized, curtailed)):
                totals[key] += value
            self.totals["pv_available_kwh"] += available
            self.totals["pv_utilized_kwh"] += utilized
            self.totals["curtailment_kwh"] += curtailed
        voltages = np.asarray([info["voltages_pu"][f"bus{i}"] for i in range(2, 7)])
        self.voltage_min_phase = np.minimum(self.voltage_min_phase, voltages.min(axis=0))
        self.voltage_max_phase = np.maximum(self.voltage_max_phase, voltages.max(axis=0))
        severity = (np.maximum(self.voltage_limits_pu[0] - voltages, 0)
                    + np.maximum(voltages - self.voltage_limits_pu[1], 0))
        violating = severity > 0
        self.voltage_violation_steps += int(violating.any())
        self.voltage_violation_steps_phase += violating.any(axis=0)
        self.voltage_violation_node_phase_count += int(violating.sum())
        self.voltage_violation_max_pu = max(self.voltage_violation_max_pu, float(severity.max()))
        self.voltage_violation_sum_pu += float(severity.sum())
        self.voltage_violation_integral_pu_h += float(severity.sum()) * dt_h
        pf = np.asarray(info["pf_phase"], dtype=float)
        self.pf_min_phase = np.minimum(self.pf_min_phase, pf)
        self.pf_max_phase = np.maximum(self.pf_max_phase, pf)
        self.pf_sum_phase += pf
        self.pf_last_phase = pf.copy()
        gaps = np.maximum(self.pf_min - pf, 0)
        self.pf_violation_steps += int((gaps > 0).any())
        self.pf_violation_steps_phase += gaps > 0
        self.pf_violation_max = max(self.pf_violation_max, float(gaps.max()))
        self.pf_violation_sum += float(gaps.sum())
        self.bess_soc_final = float(info["controls"]["bess"]["soc_after"])
        self.bess_energy_final_kwh = float(bess_energy_kwh)
        self.departures.extend(deepcopy(info["truck_departures"]))
        self.vuf_max_pct = max(self.vuf_max_pct, info["max_lv_voltage_unbalance_pct"])
        self.transformer_loading_max_pct = max(
            self.transformer_loading_max_pct, info["transformer_loading_pct"])
        self.balance_residual_max_abs_kw = max(
            self.balance_residual_max_abs_kw, abs(info["balance_residual_kw"]))

    def snapshot(self, *, include_runtime: bool = True) -> dict:
        phase_values = lambda x: None if self.steps == 0 else x.tolist()
        result = {
            **self.totals, "steps": self.steps, "complete": self.complete,
            "phase_order": ["A", "B", "C"], "voltage_buses": ["bus2", "bus3", "bus4", "bus5", "bus6"],
            "voltage_limits_pu": list(self.voltage_limits_pu),
            "voltage_min_pu_phase": phase_values(self.voltage_min_phase),
            "voltage_max_pu_phase": phase_values(self.voltage_max_phase),
            "voltage_violation_steps": self.voltage_violation_steps,
            "voltage_violation_steps_phase": self.voltage_violation_steps_phase.tolist(),
            "voltage_violation_node_phase_count": self.voltage_violation_node_phase_count,
            "voltage_violation_max_pu": self.voltage_violation_max_pu,
            "voltage_violation_sum_pu": self.voltage_violation_sum_pu,
            "voltage_violation_integral_pu_h": self.voltage_violation_integral_pu_h,
            "pf_min_phase": phase_values(self.pf_min_phase),
            "pf_max_phase": phase_values(self.pf_max_phase),
            "pf_mean_phase": phase_values(self.pf_sum_phase / max(self.steps, 1)),
            "pf_last_phase": None if self.pf_last_phase is None else self.pf_last_phase.tolist(),
            "pf_violation_steps": self.pf_violation_steps,
            "pf_violation_steps_phase": self.pf_violation_steps_phase.tolist(),
            "pf_violation_max": self.pf_violation_max, "pf_violation_sum": self.pf_violation_sum,
            "bess_soc_initial": self.bess_soc_initial, "bess_soc_final": self.bess_soc_final,
            "bess_energy_initial_kwh": self.bess_energy_initial_kwh,
            "bess_energy_final_kwh": self.bess_energy_final_kwh,
            "truck_departure_count": len(self.departures),
            "truck_departure_violation_count": sum(e["departure_violation"] for e in self.departures),
            "truck_energy_deficit_kwh": sum(e["energy_deficit_kwh"] for e in self.departures),
            "truck_departures": deepcopy(self.departures),
            "pv_by_plant": deepcopy(self.pv_by_plant),
            "pv_utilization_fraction": (self.totals["pv_utilized_kwh"] / self.totals["pv_available_kwh"]
                                        if self.totals["pv_available_kwh"] > 0 else None),
            "vuf_max_pct": self.vuf_max_pct,
            "transformer_loading_max_pct": self.transformer_loading_max_pct,
            "balance_residual_max_abs_kw": self.balance_residual_max_abs_kw,
            "solver_attempts": self.steps + self.failed_solver_attempts,
            "converged_steps": self.steps,
            "failed_solver_attempts": self.failed_solver_attempts,
            "all_solver_attempts_converged": self.failed_solver_attempts == 0,
            "last_solver_error": self.last_solver_error,
        }
        if include_runtime:
            result["runtime"] = {
                "reset_seconds": self.reset_seconds,
                "step_seconds_total": self.step_seconds_total,
                "step_seconds_max": self.step_seconds_max,
                "elapsed_wall_seconds": (self.finished_at if self.finished_at is not None
                                         else time.perf_counter()) - self.started_at,
                "note": "Observational wall time; excluded from deterministic trajectory comparisons.",
            }
        return result

