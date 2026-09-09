"""Regresiones de potencia constante explícita entre 0.80 y 1.10 pu."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import opendssdirect as dss


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import ElectricalSnapshotInputs, MicrorredIndustrial
from env.opendss_easy import DEFAULT_LOAD_VMAXPU, DEFAULT_LOAD_VMINPU, DSSCircuit


class ConstantPowerVoltageLimitTests(unittest.TestCase):
    @staticmethod
    def _controlled_load(source_pu: float, sign: float = 1.0):
        circuit = DSSCircuit(name="constant_power_controlled", basekv=0.4, phases=3)
        circuit.connect_to_grid(bus="bus2", pu=source_pu, angle=0.0, vsrc_name="Source")
        circuit.add_load("known", "bus2", phases=3, kw=sign * 50.0, kvar=sign * 30.0, model=1)
        if not circuit.solve():
            raise AssertionError("El circuito controlado no convergió.")
        row = circuit.list_loads_df().iloc[0]
        vbus = float(circuit.get_bus_voltages_df(["bus2"])["V_pu"].mean())
        return row, vbus

    def test_01_defaults_are_explicit_and_do_not_modify_vlowpu(self):
        circuit = DSSCircuit(name="explicit_voltage_limits", basekv=0.4, phases=3)
        circuit.connect_to_grid("bus2")
        circuit.add_load("known", "bus2", phases=3, kw=50.0, kvar=30.0, model=1)
        dss.Loads.Name("known")
        self.assertEqual(dss.Loads.Model(), 1)
        self.assertEqual(dss.Loads.Vminpu(), DEFAULT_LOAD_VMINPU)
        self.assertEqual(dss.Loads.Vmaxpu(), DEFAULT_LOAD_VMAXPU)
        self.assertEqual(str(dss.run_command("? Load.known.Vlowpu")), "0.5")

    def test_02_positive_load_holds_pq_down_to_approximately_080_pu(self):
        for source_pu in (1.00, 0.95, 0.93, 0.90, 0.85, 0.8001):
            row, vbus = self._controlled_load(source_pu)
            self.assertGreaterEqual(vbus, 0.80)
            self.assertAlmostEqual(float(row["resolved_kW"]), 50.0, delta=1e-5)
            self.assertAlmostEqual(float(row["resolved_kvar"]), 30.0, delta=1e-5)

    def test_03_negative_load_holds_pq_at_low_and_high_voltage(self):
        for source_pu in (0.93, 0.90, 0.85, 0.8001, 1.065):
            row, vbus = self._controlled_load(source_pu, sign=-1.0)
            self.assertGreaterEqual(vbus, 0.80)
            self.assertLessEqual(vbus, 1.10)
            self.assertAlmostEqual(float(row["resolved_kW"]), -50.0, delta=1e-5)
            self.assertAlmostEqual(float(row["resolved_kvar"]), -30.0, delta=1e-5)

    def test_04_severe_snapshot_loads_hold_their_commands(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, seed=42)
        weights = np.array([0.40, 0.40, 0.20])
        result = env._solve_electrical_snapshot(ElectricalSnapshotInputs(
            p_bus3_abc_kw=18.74 * weights + np.array([22.0, 22.0, 0.0]),
            q_bus3_abc_kvar=18.74 * math.tan(math.acos(0.80)) * weights,
            p_wh1_abc_kw=65.66 * weights,
            q_wh1_abc_kvar=65.66 * math.tan(math.acos(0.57)) * weights,
            p_wh2_abc_kw=76.12 * weights,
            q_wh2_abc_kvar=76.12 * math.tan(math.acos(0.57)) * weights,
            bess_p_inj_kw=-159.9511018778908, bess_q_inj_kvar=-82.55691980717668,
            truck5_p_inj_kw=-133.2324970655224, truck5_q_inj_kvar=-68.91372668551284,
            truck6_p_inj_kw=-132.9911315699737, truck6_q_inj_kvar=-69.3783750439425,
        ))
        self.assertTrue(result.converged)
        self.assertLess(float(np.min(result.voltages_vpu["bus5"])), 0.95)
        loads_df = result.power_balance["loads_df"]
        nonzero_p = loads_df[loads_df["assigned_kW"].abs() > 1e-8]
        nonzero_q = loads_df[loads_df["assigned_kvar"].abs() > 1e-8]
        self.assertLess(float((nonzero_p["resolved_kW"] - nonzero_p["assigned_kW"]).abs().max()), 1e-5)
        self.assertLess(float((nonzero_q["resolved_kvar"] - nonzero_q["assigned_kvar"]).abs().max()), 1e-5)


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
