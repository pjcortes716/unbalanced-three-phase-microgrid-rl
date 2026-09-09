"""Pruebas de instrumentación eléctrica diagnóstica fuera de la observación RL."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import (
    PHASE_LABELS,
    ElectricalSnapshotInputs,
    MicrorredIndustrial,
    calculate_power_factor,
    calculate_voltage_unbalance_pct,
)


REPORTED_BUSES = ("bus1", "bus2", "bus3", "bus4", "bus5", "bus6")


class ElectricalInstrumentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, seed=42)

    def _snapshot(self, **kwargs):
        return self.env._solve_electrical_snapshot(ElectricalSnapshotInputs(**kwargs))

    @staticmethod
    def _unbalanced_inputs() -> dict:
        return {
            "p_bus3_abc_kw": np.array([15.0, 5.0, 0.0]),
            "q_bus3_abc_kvar": np.array([3.0, 1.0, 0.0]),
        }

    def test_01_vuf_uses_abc_symmetric_components(self):
        self.assertEqual(PHASE_LABELS, ("A", "B", "C"))
        balanced = np.array(
            [
                1.0 * np.exp(1j * 0.0),
                1.0 * np.exp(-1j * 2.0 * np.pi / 3.0),
                1.0 * np.exp(1j * 2.0 * np.pi / 3.0),
            ]
        )
        self.assertAlmostEqual(calculate_voltage_unbalance_pct(balanced), 0.0, places=10)

        synthetic = np.array(
            [
                1.00 * np.exp(1j * 0.0),
                0.95 * np.exp(-1j * 2.0 * np.pi / 3.0),
                1.02 * np.exp(1j * 2.0 * np.pi / 3.0),
            ]
        )
        # Cálculo independiente del oráculo, sin invocar la función bajo prueba.
        a = np.exp(2j * np.pi / 3.0)
        v1_expected = (synthetic[0] + a * synthetic[1] + a**2 * synthetic[2]) / 3.0
        v2_expected = (synthetic[0] + a**2 * synthetic[1] + a * synthetic[2]) / 3.0
        expected_pct = 100.0 * abs(v2_expected) / abs(v1_expected)
        self.assertAlmostEqual(
            calculate_voltage_unbalance_pct(synthetic), expected_pct, places=12
        )

    def test_02_power_factor_is_unsigned_and_robust_at_zero_power(self):
        expected_export = 10.0 / np.hypot(10.0, 5.0)
        self.assertAlmostEqual(calculate_power_factor(-10.0, 5.0), expected_export, places=12)
        self.assertGreaterEqual(calculate_power_factor(-10.0, 5.0), 0.0)
        self.assertAlmostEqual(calculate_power_factor(0.0, 5.0), 0.0, places=12)
        self.assertEqual(calculate_power_factor(0.0, 0.0), 1.0)

    def test_03_balanced_snapshot_reports_near_zero_vuf_and_manual_pf(self):
        result = self._snapshot(
            p_wh1_abc_kw=np.full(3, 50.0),
            q_wh1_abc_kvar=np.full(3, 100.0 / 3.0),
        )
        self.assertTrue(result.converged)
        self.assertEqual(tuple(result.voltages_vpu), REPORTED_BUSES)
        self.assertLess(result.max_voltage_unbalance_pct, 1e-5)
        np.testing.assert_allclose(
            result.pcc_pf_phase,
            [
                abs(p_kw) / np.hypot(p_kw, q_kvar)
                for p_kw, q_kvar in zip(result.pcc_p_kw_phase, result.pcc_q_kvar_phase)
            ],
            atol=1e-12,
        )
        self.assertLess(np.ptp(result.pcc_pf_phase), 1e-10)
        self.assertAlmostEqual(
            result.pcc_pf_total,
            abs(result.pcc["P_total_kW"])
            / np.hypot(result.pcc["P_total_kW"], result.pcc["Q_total_kvar"]),
            places=12,
        )

        info = result.electrical_info()
        self.assertTrue(info["electrical_converged"])
        self.assertEqual(info["voltage_phase_order"], ["A", "B", "C"])
        self.assertEqual(tuple(info["voltages_pu"]), REPORTED_BUSES)
        self.assertTrue(all(isinstance(values, list) for values in info["voltages_pu"].values()))
        np.testing.assert_allclose(info["pcc_pf_phase"], result.pcc_pf_phase, atol=1e-12)

    def test_04_unbalanced_snapshot_has_spatial_vuf(self):
        result = self._snapshot(**self._unbalanced_inputs())
        self.assertTrue(result.converged)
        self.assertEqual(tuple(result.voltage_unbalance_pct), REPORTED_BUSES)
        lv_vuf = np.array([result.voltage_unbalance_pct[bus] for bus in REPORTED_BUSES[1:]])
        self.assertGreater(result.voltage_unbalance_pct["bus3"], 0.01)
        self.assertGreater(lv_vuf.max(), result.voltage_unbalance_pct["bus1"] * 25.0)
        self.assertGreater(np.ptp(lv_vuf), 1e-3)
        self.assertAlmostEqual(result.max_voltage_unbalance_pct, lv_vuf.max(), places=12)
        self.assertIn(result.worst_unbalance_bus, REPORTED_BUSES[1:])
        self.assertAlmostEqual(result.max_lv_voltage_unbalance_pct, lv_vuf.max(), places=12)

    def test_05_voltage_extrema_include_bus_and_abc_phase(self):
        result = self._snapshot(**self._unbalanced_inputs())
        extrema = [
            (float(value), bus, PHASE_LABELS[phase_index])
            for bus, values in result.voltages_vpu.items()
            for phase_index, value in enumerate(values)
        ]
        expected_min = min(extrema, key=lambda item: item[0])
        expected_max = max(extrema, key=lambda item: item[0])
        self.assertAlmostEqual(result.vmin_pu, expected_min[0], places=12)
        self.assertEqual((result.vmin_bus, result.vmin_phase), expected_min[1:])
        self.assertAlmostEqual(result.vmax_pu, expected_max[0], places=12)
        self.assertEqual((result.vmax_bus, result.vmax_phase), expected_max[1:])

    def test_06_real_reset_snapshot_is_diagnostic_and_observation_stays_114d(self):
        obs, _ = self.env.reset(seed=42)
        obs_before = self.env._get_obs().copy()
        time_before = self.env.t
        hour_state = self.env._collect_hour_state(self.env.t)
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
        info = result.electrical_info()

        self.assertTrue(result.converged)
        self.assertEqual(set(info["voltages_pu"]), set(REPORTED_BUSES))
        self.assertTrue(np.isfinite(result.pcc_p_kw_phase).all())
        self.assertTrue(np.isfinite(result.pcc_q_kvar_phase).all())
        self.assertTrue(np.isfinite(result.pcc_pf_phase).all())
        self.assertGreaterEqual(result.transformer_loading_percent, 0.0)
        self.assertEqual(self.env.t, time_before)
        self.assertEqual(obs.shape, (114,))
        self.assertEqual(self.env.observation_space.shape, (114,))
        np.testing.assert_array_equal(self.env._get_obs(), obs_before)


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
