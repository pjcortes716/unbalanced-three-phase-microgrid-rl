from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import MicrorredIndustrial
from env.battery import Battery
from env.v2g_bus_5 import V2GTruck


class PVAndActionDecoder2DTests(unittest.TestCase):
    def _env_at_solar_hour(self, seed=42):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.reset(seed=seed)
        t = next(
            hour
            for hour in range(env.horizon)
            if all(plant.available_at(hour) > 0.0 for plant in (env.pv1, env.pv2, env.pv3))
        )
        return env, t

    def test_all_pv_share_one_weather_week_and_keep_individual_pmax(self):
        env, _ = self._env_at_solar_hour()
        self.assertEqual(env.pv1.sampler.week_idx, env.pv_week_idx)
        self.assertEqual(env.pv2.sampler.week_idx, env.pv_week_idx)
        self.assertEqual(env.pv3.sampler.week_idx, env.pv_week_idx)
        self.assertEqual(env.pv1.sampler.pmax_target_kw, 74.25)
        self.assertEqual(env.pv2.sampler.pmax_target_kw, 75.58)
        self.assertEqual(env.pv3.sampler.pmax_target_kw, 7.75)
        self.assertEqual(env.pv1.s_max_kva, 100.0)
        self.assertEqual(env.pv2.s_max_kva, 100.0)
        self.assertEqual(env.pv3.s_max_kva, 10.0)

        # Misma meteorología base, con escala nominal distinta. La potencia
        # real incorpora una pequeña perturbación espacial por planta.
        np.testing.assert_allclose(
            env.pv1.sampler.P_base_week / 74.25,
            env.pv2.sampler.P_base_week / 75.58,
        )
        np.testing.assert_allclose(
            env.pv1.sampler.P_base_week / 74.25,
            env.pv3.sampler.P_base_week / 7.75,
        )

    def test_pv_week_is_reproducible_and_changes_with_a_different_seed(self):
        env_a = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        _, info_a = env_a.reset(seed=42)
        env_b = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        _, info_b = env_b.reset(seed=42)
        self.assertEqual(info_a["pv_week_idx"], info_b["pv_week_idx"])
        self.assertEqual(info_a["week_indices"]["pv1"], info_a["week_indices"]["pv2"])
        self.assertEqual(info_a["week_indices"]["pv2"], info_a["week_indices"]["pv3"])

        env_c = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        _, info_c = env_c.reset(seed=456)
        self.assertNotEqual(info_a["pv_week_idx"], info_c["pv_week_idx"])

    def test_decoder_pv_active_extremes_q_and_capability(self):
        env, t = self._env_at_solar_hour()
        cases = ((1.0, 0.0, 1.0), (-1.0, 1.0, 0.0), (0.0, 0.5, 0.5))
        pv_specs = (
            ("pv1", env.ACTION_PV1_UTILIZATION, env.ACTION_PV1_Q, env.pv1.s_max_kva),
            ("pv2", env.ACTION_PV2_UTILIZATION, env.ACTION_PV2_Q, env.pv2.s_max_kva),
            ("pv3", env.ACTION_PV3_UTILIZATION, env.ACTION_PV3_Q, env.pv3.s_max_kva),
        )
        for name, active_idx, q_idx, s_max in pv_specs:
            for external_active, expected_curtailment, expected_utilization in cases:
                action = np.zeros(12, dtype=float)
                action[active_idx] = external_active
                decoded = env._decode_action(action, t)
                pv = decoded[name]
                self.assertAlmostEqual(pv["utilization"], expected_utilization)
                self.assertAlmostEqual(pv["curtailment"], expected_curtailment)
                self.assertAlmostEqual(
                    pv["p_command_kw"], expected_utilization * pv["p_available_kw"]
                )

            for q_external in (-1.0, 1.0):
                action = np.zeros(12, dtype=float)
                action[active_idx] = 0.0
                action[q_idx] = q_external
                pv = env._decode_action(action, t)[name]
                self.assertAlmostEqual(pv["q_command_kvar"], q_external * pv["q_cap_kvar"])
                self.assertLessEqual(
                    pv["p_command_kw"] ** 2 + pv["q_command_kvar"] ** 2,
                    s_max**2 + 1e-10,
                )

    def test_decoder_bess_and_truck_transformations_and_zero_action(self):
        env, t = self._env_at_solar_hour()
        positive = np.zeros(12)
        positive[env.ACTION_BESS_P] = 1.0
        positive[env.ACTION_BESS_Q] = 0.5
        positive[env.ACTION_TRUCK5_P] = 0.2
        positive[env.ACTION_TRUCK5_Q] = 0.5
        positive[env.ACTION_TRUCK6_P] = -0.2
        positive[env.ACTION_TRUCK6_Q] = -0.5
        decoded = env._decode_action(positive, t)
        self.assertEqual(decoded["bess"]["p_requested_inj_kw"], 180.0)
        self.assertEqual(decoded["bess"]["u_internal_kw"], -180.0)
        self.assertEqual(decoded["bess"]["q_frac"], 0.5)
        self.assertEqual(decoded["truck5"]["aP_internal"], -0.2)
        self.assertEqual(decoded["truck5"]["aQ_internal"], 0.5)
        self.assertEqual(decoded["truck6"]["aP_internal"], 0.2)
        self.assertEqual(decoded["truck6"]["aQ_internal"], -0.5)

        negative = np.zeros(12)
        negative[env.ACTION_BESS_P] = -1.0
        decoded_negative = env._decode_action(negative, t)
        self.assertEqual(decoded_negative["bess"]["p_requested_inj_kw"], -180.0)
        self.assertEqual(decoded_negative["bess"]["u_internal_kw"], 180.0)

        zero = env._decode_action(np.zeros(12), t)
        for name in ("pv1", "pv2", "pv3"):
            self.assertEqual(zero[name]["utilization"], 0.5)
        self.assertEqual(zero["bess"]["p_requested_inj_kw"], 0.0)
        self.assertEqual(zero["bess"]["q_frac"], 0.0)
        self.assertEqual(zero["truck5"]["aP_internal"], 0.0)
        self.assertTrue(env.action_space.contains(np.zeros(12, dtype=np.float32)))
        self.assertTrue(env.action_space.contains(env.action_space.sample()))

    def test_decoder_clips_outside_values_and_rejects_bad_actions(self):
        env, t = self._env_at_solar_hour()
        action = np.full(12, 1.0001)
        decoded = env._decode_action(action, t)
        self.assertTrue(all(value == 1.0 for value in decoded["action_clipped"]))
        self.assertEqual(decoded["pv1"]["utilization"], 1.0)
        with self.assertRaises(ValueError):
            env._decode_action(np.zeros(11), t)
        with self.assertRaises(ValueError):
            env._decode_action(np.full(12, np.nan), t)
        with self.assertRaises(ValueError):
            env._decode_action(np.full(12, np.inf), t)

    def test_decoder_is_pure_and_idempotent(self):
        env, t = self._env_at_solar_hour()
        action = np.array([1, 1, 0, 0, -1, -1, .2, .5, .2, .5, -.2, -.5])
        snapshot = {
            "t": env.t,
            "bess_energy": env.bess.E_kWh,
            "zoe_soc": env.zoe_fleet.soc.copy(),
            "zoe_eac": env.zoe_fleet.Eac_rem.copy(),
            "zoe_t": env.zoe_fleet.t,
            "truck5": (env.truck5.t, env.truck5.soc, env.truck5.connected),
            "truck6": (env.truck6.t, env.truck6.soc, env.truck6.connected),
            "pv": (env.pv1.t, env.pv2.t, env.pv3.t),
            "samplers": (env.office_sampler.t, env.wh1_sampler.t, env.wh2_sampler.t,
                         env.pv1.sampler.t, env.pv2.sampler.t, env.pv3.sampler.t),
        }
        decoded_a = env._decode_action(action, t)
        decoded_b = env._decode_action(action, t)
        self.assertEqual(decoded_a, decoded_b)
        self.assertEqual(env.t, snapshot["t"])
        self.assertEqual(env.bess.E_kWh, snapshot["bess_energy"])
        np.testing.assert_array_equal(env.zoe_fleet.soc, snapshot["zoe_soc"])
        np.testing.assert_array_equal(env.zoe_fleet.Eac_rem, snapshot["zoe_eac"])
        self.assertEqual(env.zoe_fleet.t, snapshot["zoe_t"])
        self.assertEqual((env.truck5.t, env.truck5.soc, env.truck5.connected), snapshot["truck5"])
        self.assertEqual((env.truck6.t, env.truck6.soc, env.truck6.connected), snapshot["truck6"])
        self.assertEqual((env.pv1.t, env.pv2.t, env.pv3.t), snapshot["pv"])
        self.assertEqual(
            (env.office_sampler.t, env.wh1_sampler.t, env.wh2_sampler.t,
             env.pv1.sampler.t, env.pv2.sampler.t, env.pv3.sampler.t),
            snapshot["samplers"],
        )

    def test_battery_signs_q_and_capability_are_empirically_consistent(self):
        discharge = Battery()
        discharge.reset(soc=0.5)
        e_before = discharge.E_kWh
        out_discharge = discharge.step(u_kW=-36.0, q_frac=0.5)
        self.assertGreater(out_discharge["P_grid_kW"], 0.0)
        self.assertGreater(out_discharge["Q_set_kvar"], 0.0)
        self.assertLess(out_discharge["E_kWh"], e_before)
        self.assertLessEqual(out_discharge["S_used_kVA"] ** 2, discharge.S_rated_kVA**2 + 1e-10)

        charge = Battery()
        charge.reset(soc=0.5)
        e_before = charge.E_kWh
        out_charge = charge.step(u_kW=36.0, q_frac=-0.5)
        self.assertLess(out_charge["P_grid_kW"], 0.0)
        self.assertLess(out_charge["Q_set_kvar"], 0.0)
        self.assertGreater(out_charge["E_kWh"], e_before)
        self.assertLessEqual(out_charge["S_used_kVA"] ** 2, charge.S_rated_kVA**2 + 1e-10)

    def _connected_truck(self) -> V2GTruck:
        truck = V2GTruck(np.ones((7, 24), dtype=int), seed=1)
        truck.reset(seed=1)
        truck.soc = 0.5
        return truck

    def test_v2g_external_signs_q_and_capability(self):
        # Descarga: aP interno negativo; P_grid legacy negativo, por lo que
        # la inyección física que se usará más adelante es -P_grid positiva.
        discharge = self._connected_truck()
        soc_before = discharge.soc
        p_grid, q_grid = discharge.apply_action_at(0, aP=-0.2, aQ=0.5)
        self.assertLess(p_grid, 0.0)
        self.assertGreater(-p_grid, 0.0)
        self.assertGreater(q_grid, 0.0)
        self.assertLess(discharge.soc, soc_before)
        self.assertLessEqual(p_grid**2 + q_grid**2, discharge.s_max**2 + 1e-10)

        charge = self._connected_truck()
        soc_before = charge.soc
        p_grid, q_grid = charge.apply_action_at(0, aP=0.2, aQ=-0.5)
        self.assertGreater(p_grid, 0.0)
        self.assertLess(-p_grid, 0.0)
        self.assertLess(q_grid, 0.0)
        self.assertGreater(charge.soc, soc_before)
        self.assertLessEqual(p_grid**2 + q_grid**2, charge.s_max**2 + 1e-10)

    def test_v2g_disconnected_ignores_commands_without_soc_change(self):
        truck = V2GTruck(np.zeros((7, 24), dtype=int), seed=2)
        truck.reset(seed=2)
        soc_before = truck.soc
        p_grid, q_grid = truck.apply_action_at(0, aP=-1.0, aQ=1.0)
        self.assertEqual(p_grid, 0.0)
        self.assertEqual(q_grid, 0.0)
        self.assertEqual(truck.soc, soc_before)
        self.assertLessEqual(p_grid**2 + q_grid**2, truck.s_max**2 + 1e-10)


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
