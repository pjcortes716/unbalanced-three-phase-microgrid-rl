"""Pruebas de ETAPA 2E: snapshot eléctrico puro de una hora."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import ElectricalSnapshotInputs, MicrorredIndustrial


class ElectricalSnapshot2ETests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, seed=42)

    def _snapshot(self, **kwargs):
        return self.env._solve_electrical_snapshot(ElectricalSnapshotInputs(**kwargs))

    @staticmethod
    def _balanced_bus3(power_per_phase_kw: float = 10.0) -> dict:
        return {"p_bus3_abc_kw": np.full(3, power_per_phase_kw, dtype=float)}

    def test_01_all_zero(self):
        result = self._snapshot()
        self.assertTrue(result.converged)
        self.assertLess(abs(result.pcc["P_total_kW"]), 1e-6)
        self.assertLess(abs(result.p_losses_kw), 1e-6)
        for values in result.voltages_vpu.values():
            np.testing.assert_allclose(values, 1.0, atol=1e-6)

    def test_02_simple_balanced_load(self):
        result = self._snapshot(**self._balanced_bus3())
        self.assertTrue(result.converged)
        self.assertAlmostEqual(
            result.pcc["P_total_kW"], 30.0 + result.p_losses_kw, delta=2e-4
        )
        phase_p = np.array([result.pcc[key] for key in ("P_a_kW", "P_b_kW", "P_c_kW")])
        self.assertLess(np.ptp(phase_p), 1e-4)
        self.assertLess(np.ptp(result.voltages_vpu["bus3"]), 1e-6)

    def test_03_unbalanced_active_load(self):
        result = self._snapshot(p_bus3_abc_kw=np.array([15.0, 5.0, 0.0]))
        self.assertTrue(result.converged)
        self.assertAlmostEqual(
            result.pcc["P_total_kW"], 20.0 + result.p_losses_kw, delta=2e-4
        )
        phase_p = np.array([result.pcc[key] for key in ("P_a_kW", "P_b_kW", "P_c_kW")])
        self.assertGreater(np.ptp(phase_p), 1.0)
        self.assertGreater(np.ptp(result.voltages_vpu["bus3"]), 1e-7)

    def test_04_unbalanced_reactive_load(self):
        result = self._snapshot(
            p_bus3_abc_kw=np.array([6.0, 6.0, 6.0]),
            q_bus3_abc_kvar=np.array([4.0, 2.0, 1.0]),
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(
            result.pcc["Q_total_kvar"], 7.0 + result.q_losses_kvar, delta=2e-4
        )
        phase_q = np.array([result.pcc[key] for key in ("Q_a_kvar", "Q_b_kvar", "Q_c_kvar")])
        self.assertGreater(np.ptp(phase_q), 1.0)
        self.assertTrue(np.isfinite(result.voltages_vpu["bus3"]).all())

    def test_05_balanced_pv_reduces_import(self):
        baseline = self._snapshot(**self._balanced_bus3())
        with_pv = self._snapshot(**self._balanced_bus3(), pv3_p_inj_kw=15.0)
        self.assertTrue(with_pv.converged)
        self.assertAlmostEqual(
            baseline.pcc["P_total_kW"] - with_pv.pcc["P_total_kW"], 15.0, delta=0.1
        )

    def test_06_export(self):
        result = self._snapshot(**self._balanced_bus3(), pv1_p_inj_kw=50.0)
        self.assertTrue(result.converged)
        self.assertLess(result.pcc["P_total_kW"], 0.0)

    def test_07_bess_charge_and_discharge(self):
        baseline = self._snapshot(**self._balanced_bus3())
        discharge = self._snapshot(**self._balanced_bus3(), bess_p_inj_kw=20.0)
        charge = self._snapshot(**self._balanced_bus3(), bess_p_inj_kw=-20.0)
        self.assertAlmostEqual(
            baseline.pcc["P_total_kW"] - discharge.pcc["P_total_kW"], 20.0, delta=0.1
        )
        self.assertAlmostEqual(
            charge.pcc["P_total_kW"] - baseline.pcc["P_total_kW"], 20.0, delta=0.1
        )

    def test_08_truck_charge_and_discharge(self):
        baseline = self._snapshot(**self._balanced_bus3())
        discharge = self._snapshot(**self._balanced_bus3(), truck5_p_inj_kw=15.0)
        charge = self._snapshot(**self._balanced_bus3(), truck5_p_inj_kw=-15.0)
        self.assertAlmostEqual(
            baseline.pcc["P_total_kW"] - discharge.pcc["P_total_kW"], 15.0, delta=0.1
        )
        self.assertAlmostEqual(
            charge.pcc["P_total_kW"] - baseline.pcc["P_total_kW"], 15.0, delta=0.1
        )

    def test_09_inverter_reactive_signs(self):
        baseline = self._snapshot(**self._balanced_bus3())
        q_injection = self._snapshot(**self._balanced_bus3(), pv1_q_inj_kvar=20.0)
        q_absorption = self._snapshot(**self._balanced_bus3(), pv1_q_inj_kvar=-20.0)
        self.assertLess(q_injection.pcc["Q_total_kvar"], baseline.pcc["Q_total_kvar"])
        self.assertGreater(q_absorption.pcc["Q_total_kvar"], baseline.pcc["Q_total_kvar"])

    def test_10_physical_power_balance(self):
        result = self._snapshot(
            p_bus3_abc_kw=np.array([15.0, 5.0, 0.0]),
            q_bus3_abc_kvar=np.array([3.0, 1.0, 0.0]),
            p_wh1_abc_kw=np.array([4.0, 4.0, 4.0]),
            q_wh1_abc_kvar=np.array([1.0, 1.0, 1.0]),
            pv1_p_inj_kw=10.0,
            bess_p_inj_kw=5.0,
            truck6_p_inj_kw=-2.0,
        )
        self.assertTrue(result.converged)
        balance = result.power_balance
        manual = (
            result.pcc["P_total_kW"]
            + balance["P_local_injection_kW"]
            - balance["P_load_total_kW"]
            - result.p_losses_kw
        )
        self.assertAlmostEqual(manual, result.balance_residual_kw, delta=1e-12)
        self.assertLess(abs(result.balance_residual_kw), 2e-5)
        self.assertLess(abs(balance["balance_residual_kW"]), 2e-5)

    def test_11_three_phase_total_power_is_not_tripled(self):
        result = self._snapshot(pv1_p_inj_kw=30.0)
        self.assertTrue(result.converged)
        phase_power = result.element_powers["pv1"]["P_inj_abc_kW"]
        np.testing.assert_allclose(phase_power, np.full(3, 10.0), atol=1e-3)
        self.assertAlmostEqual(float(phase_power.sum()), 30.0, delta=1e-3)

    def test_12_consecutive_snapshots_are_isolated(self):
        first = self._snapshot(p_bus3_abc_kw=np.array([30.0, 0.0, 0.0]))
        second = self._snapshot()
        self.assertTrue(first.converged)
        self.assertTrue(second.converged)
        self.assertLess(abs(second.pcc["P_total_kW"]), 1e-6)
        self.assertLess(abs(second.p_losses_kw), 1e-6)
        for values in second.voltages_vpu.values():
            np.testing.assert_allclose(values, 1.0, atol=1e-6)

    def test_13_real_reset_hour_and_purity(self):
        self.env.reset(seed=42)
        obs_before = self.env._get_obs().copy()
        profiles_before = tuple(
            plant.P_avail_week.copy() for plant in (self.env.pv1, self.env.pv2, self.env.pv3)
        )
        rng_states_before = tuple(
            deepcopy(rng.bit_generator.state)
            for rng in (
                self.env.np_random,
                self.env.office_sampler.rng,
                self.env.wh1_sampler.rng,
                self.env.wh2_sampler.rng,
                self.env.pv1.sampler.rng,
                self.env.pv2.sampler.rng,
                self.env.pv3.sampler.rng,
                self.env.truck5.rng,
                self.env.truck6.rng,
                self.env.zoe_fleet.rng,
            )
        )
        state_before = {
            "t": self.env.t,
            "bess": (self.env.bess.E_kWh, self.env.bess.soc),
            "truck5": (self.env.truck5.t, self.env.truck5.soc, self.env.truck5.P_grid_kw),
            "truck6": (self.env.truck6.t, self.env.truck6.soc, self.env.truck6.P_grid_kw),
            "zoe_soc": self.env.zoe_fleet.soc.copy(),
            "zoe_energy": self.env.zoe_fleet.Eac_rem.copy(),
            "zoe_t": self.env.zoe_fleet.t,
            "sampler_t": (
                self.env.office_sampler.t, self.env.wh1_sampler.t, self.env.wh2_sampler.t,
                self.env.pv1.t, self.env.pv2.t, self.env.pv3.t,
            ),
        }

        # t=0 es una hora real y preparada por reset(); se incluyen Zoe en
        # bus3 y la disponibilidad PV real, con BESS/trucks en cero.
        hour = self.env.t
        hour_state = self.env._collect_hour_state(hour)
        result = self.env._solve_electrical_snapshot(
            ElectricalSnapshotInputs(
                p_bus3_abc_kw=hour_state["p_bus3_abc"],
                q_bus3_abc_kvar=hour_state["q_bus3_abc"],
                p_wh1_abc_kw=hour_state["p_wh1_abc"],
                q_wh1_abc_kvar=hour_state["q_wh1_abc"],
                p_wh2_abc_kw=hour_state["p_wh2_abc"],
                q_wh2_abc_kvar=hour_state["q_wh2_abc"],
                pv1_p_inj_kw=hour_state["pv_available"][0],
                pv2_p_inj_kw=hour_state["pv_available"][1],
                pv3_p_inj_kw=hour_state["pv_available"][2],
            )
        )

        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.pcc["P_total_kW"]))
        self.assertTrue(np.isfinite(result.p_losses_kw))
        self.assertLess(abs(result.balance_residual_kw), 5e-5)
        self.assertTrue(all(np.isfinite(v).all() for v in result.voltages_vpu.values()))

        self.assertEqual(self.env.t, state_before["t"])
        self.assertEqual((self.env.bess.E_kWh, self.env.bess.soc), state_before["bess"])
        self.assertEqual(
            (self.env.truck5.t, self.env.truck5.soc, self.env.truck5.P_grid_kw),
            state_before["truck5"],
        )
        self.assertEqual(
            (self.env.truck6.t, self.env.truck6.soc, self.env.truck6.P_grid_kw),
            state_before["truck6"],
        )
        np.testing.assert_array_equal(self.env.zoe_fleet.soc, state_before["zoe_soc"])
        np.testing.assert_array_equal(self.env.zoe_fleet.Eac_rem, state_before["zoe_energy"])
        self.assertEqual(self.env.zoe_fleet.t, state_before["zoe_t"])
        self.assertEqual(
            (
                self.env.office_sampler.t, self.env.wh1_sampler.t, self.env.wh2_sampler.t,
                self.env.pv1.t, self.env.pv2.t, self.env.pv3.t,
            ),
            state_before["sampler_t"],
        )
        for before, plant in zip(profiles_before, (self.env.pv1, self.env.pv2, self.env.pv3)):
            np.testing.assert_array_equal(plant.P_avail_week, before)
        for expected, rng in zip(
            rng_states_before,
            (
                self.env.np_random,
                self.env.office_sampler.rng,
                self.env.wh1_sampler.rng,
                self.env.wh2_sampler.rng,
                self.env.pv1.sampler.rng,
                self.env.pv2.sampler.rng,
                self.env.pv3.sampler.rng,
                self.env.truck5.rng,
                self.env.truck6.rng,
                self.env.zoe_fleet.rng,
            ),
        ):
            self.assertEqual(rng.bit_generator.state, expected)
        np.testing.assert_array_equal(self.env._get_obs(), obs_before)


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
