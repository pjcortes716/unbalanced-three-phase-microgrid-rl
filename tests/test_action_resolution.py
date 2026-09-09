from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.battery import Battery
from env.v2g_bus_5 import V2GTruck


class BatteryActionResolutionTests(unittest.TestCase):
    @staticmethod
    def _battery(energy_kwh: float) -> Battery:
        battery = Battery()
        battery.reset(E_init_kWh=energy_kwh)
        return battery

    @staticmethod
    def _battery_state(battery: Battery):
        return (
            battery.E_kWh,
            battery.p_plus_kW,
            battery.p_minus_kW,
            battery.Q_set_kvar,
            battery.last_P_grid_kW,
            battery.last_S_used_kVA,
            battery.last_pf_out,
        )

    def test_preview_is_pure_and_repeatable(self):
        battery = self._battery(125.0)
        before = self._battery_state(battery)

        first = battery.resolve_action(-36.0, dt_h=1.0, q_frac=0.5)
        second = battery.resolve_action(-36.0, dt_h=1.0, q_frac=0.5)

        self.assertEqual(first, second)
        self.assertEqual(before, self._battery_state(battery))
        self.assertGreater(first.P_grid_kW, 0.0)
        self.assertGreater(first.Q_set_kvar, 0.0)

    def test_resolution_commit_matches_legacy_step_at_operating_limits(self):
        cases = (
            ("normal_charge", 125.0, 36.0, 0.25),
            ("normal_discharge", 125.0, -36.0, -0.25),
            ("charge_soc_limit", 244.5, 180.0, 0.0),
            ("discharge_soc_limit", 20.5, -180.0, 0.0),
            ("charge_power_limit", 20.0, 1000.0, 0.0),
            ("discharge_power_limit", 245.0, -1000.0, 0.0),
            ("positive_q_s_limit", 125.0, -144.0, 1.0),
            ("negative_q_s_limit", 125.0, -144.0, -1.0),
            ("zero", 125.0, 0.0, 0.0),
        )
        keys = (
            "E_kWh", "soc", "p_plus_kW", "p_minus_kW", "P_grid_kW",
            "Q_set_kvar", "S_used_kVA", "pf_out",
        )

        for name, energy_kwh, u_kw, q_frac in cases:
            with self.subTest(name=name):
                resolved = self._battery(energy_kwh)
                legacy = self._battery(energy_kwh)

                preview = resolved.resolve_action(u_kw, dt_h=1.0, q_frac=q_frac)
                committed = resolved.commit_resolution(preview)
                stepped = legacy.step(u_kw, dt_h=1.0, q_frac=q_frac)

                for key in keys:
                    self.assertAlmostEqual(committed[key], stepped[key])
                self.assertAlmostEqual(resolved.E_kWh, legacy.E_kWh)
                self.assertLessEqual(preview.S_used_kVA, resolved.S_rated_kVA + 1e-10)

        self.assertAlmostEqual(
            self._battery(20.0).resolve_action(1000.0).p_plus_kW,
            180.0,
        )
        self.assertAlmostEqual(
            self._battery(245.0).resolve_action(-1000.0).p_minus_kW,
            180.0,
        )

    def test_commit_rejects_stale_preview(self):
        battery = self._battery(125.0)
        preview = battery.resolve_action(36.0)
        battery.E_kWh += 0.01

        with self.assertRaisesRegex(RuntimeError, "desactualizada"):
            battery.commit_resolution(preview)


class V2GActionResolutionTests(unittest.TestCase):
    @staticmethod
    def _truck(connected: bool, soc: float = 0.5) -> V2GTruck:
        availability = np.ones((7, 24), dtype=int) if connected else np.zeros((7, 24), dtype=int)
        truck = V2GTruck(availability, seed=7)
        truck.reset(seed=7)
        truck.soc = soc
        return truck

    @staticmethod
    def _truck_state(truck: V2GTruck):
        return (
            truck.t,
            truck.connected,
            truck.soc,
            truck.P_grid_kw,
            truck.Q_grid_kvar,
            truck.last_qmax,
            truck._action_applied_t,
        )

    def test_preview_is_pure_and_repeatable(self):
        truck = self._truck(connected=True)
        before = self._truck_state(truck)

        first = truck.resolve_action_at(0, aP=-0.2, aQ=0.5)
        second = truck.resolve_action_at(0, aP=-0.2, aQ=0.5)

        self.assertEqual(first, second)
        self.assertEqual(before, self._truck_state(truck))
        self.assertLess(first.P_grid_kw, 0.0)
        self.assertGreater(first.Q_grid_kvar, 0.0)

    def test_preview_does_not_apply_and_legacy_api_still_rejects_double_action(self):
        truck = self._truck(connected=True)
        truck.resolve_action_at(0, aP=0.2, aQ=-0.5)

        truck.apply_action_at(0, aP=0.2, aQ=-0.5)
        with self.assertRaisesRegex(RuntimeError, "Ya se aplicó"):
            truck.apply_action_at(0, aP=0.0, aQ=0.0)

    def test_resolution_commit_advance_matches_legacy_across_limits(self):
        cases = (
            ("normal_charge", True, 0.5, 0.2, -0.5),
            ("normal_discharge", True, 0.5, -0.2, 0.5),
            ("charge_soc_limit", True, 0.999, 1.0, 0.0),
            ("discharge_soc_limit", True, 0.101, -1.0, 0.0),
            ("action_and_power_limit", True, 0.5, 2.0, -2.0),
            ("positive_q_s_limit", True, 0.5, 0.8, 1.0),
            ("negative_q_s_limit", True, 0.5, 0.8, -1.0),
            ("zero", True, 0.5, 0.0, 0.0),
            ("disconnected", False, 0.5, -1.0, 1.0),
        )

        for name, connected, soc, a_p, a_q in cases:
            with self.subTest(name=name):
                resolved = self._truck(connected=connected, soc=soc)
                legacy = self._truck(connected=connected, soc=soc)

                preview = resolved.resolve_action_at(0, a_p, a_q)
                p_q_resolved = resolved.commit_action_at(0, preview)
                closing_resolved = resolved.advance_interval(0)
                p_q_legacy = legacy.apply_action_at(0, a_p, a_q)
                closing_legacy = legacy.advance_interval(0)

                np.testing.assert_allclose(p_q_resolved, p_q_legacy)
                np.testing.assert_allclose(closing_resolved, closing_legacy)
                self.assertEqual(self._truck_state(resolved), self._truck_state(legacy))
                self.assertLessEqual(
                    preview.P_grid_kw**2 + preview.Q_grid_kvar**2,
                    resolved.s_max**2 + 1e-10,
                )

    def test_preview_limits_and_disconnected_result(self):
        self.assertAlmostEqual(
            self._truck(True).resolve_action_at(0, 2.0, 0.0).P_grid_kw,
            150.0,
        )
        self.assertAlmostEqual(
            self._truck(True, 0.999).resolve_action_at(0, 1.0, 0.0).soc_next,
            1.0,
        )
        self.assertAlmostEqual(
            self._truck(True, 0.101).resolve_action_at(0, -1.0, 0.0).soc_next,
            0.10,
        )
        self.assertAlmostEqual(
            self._truck(True).resolve_action_at(0, 0.8, 1.0).Q_grid_kvar,
            90.0,
        )
        self.assertAlmostEqual(
            self._truck(True).resolve_action_at(0, 0.8, -1.0).Q_grid_kvar,
            -90.0,
        )

        disconnected = self._truck(False)
        preview = disconnected.resolve_action_at(0, -1.0, 1.0)
        self.assertEqual((preview.P_grid_kw, preview.Q_grid_kvar), (0.0, 0.0))
        self.assertEqual(preview.soc_next, disconnected.soc)

    def test_advance_does_not_apply_energy_a_second_time(self):
        truck = self._truck(connected=True, soc=0.5)
        preview = truck.resolve_action_at(0, aP=0.2, aQ=0.0)
        truck.commit_action_at(0, preview)
        soc_after_commit = truck.soc

        truck.advance_interval(0)

        self.assertAlmostEqual(truck.soc, soc_after_commit)

    def test_commit_rejects_stale_preview(self):
        truck = self._truck(connected=True)
        preview = truck.resolve_action_at(0, aP=0.2, aQ=0.0)
        truck.soc += 0.01

        with self.assertRaisesRegex(RuntimeError, "desactualizada"):
            truck.commit_action_at(0, preview)


if __name__ == "__main__":
    unittest.main()
