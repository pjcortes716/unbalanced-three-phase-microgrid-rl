from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import MicrorredIndustrial


class EnvironmentReset2CTests(unittest.TestCase):
    def _new_reset(self, seed=42):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        obs, info = env.reset(seed=seed)
        return env, obs, info

    def test_reset_returns_finite_float32_observation_inside_space(self):
        env, obs, _ = self._new_reset()
        self.assertEqual(obs.shape, (114,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertEqual(env.observation_space.shape, (114,))
        self.assertTrue(env.observation_space.contains(obs))

    def test_layout_is_contiguous_and_exactly_114(self):
        env, _, _ = self._new_reset()
        previous_stop = 0
        expected_sizes = {
            "time": 2,
            "bus3_uncontrollable_load": 6,
            "warehouse1_load": 6,
            "warehouse2_load": 6,
            "pv_available": 3,
            "bess_soc": 1,
            "forecast_office": 11,
            "forecast_wh1": 11,
            "forecast_wh2": 11,
            "forecast_pv1": 11,
            "forecast_pv2": 11,
            "forecast_pv3": 11,
            "truck5": 5,
            "truck6": 5,
            "zoe_soc": 2,
            "current_price": 1,
            "forecast_price": 11,
        }
        for name, obs_slice in env.OBS_LAYOUT.items():
            self.assertEqual(obs_slice.start, previous_stop)
            self.assertEqual(obs_slice.stop - obs_slice.start, expected_sizes[name])
            previous_stop = obs_slice.stop
        self.assertEqual(previous_stop, env.n_obs)

    def test_time_zero_and_bess_initial_soc(self):
        env, obs, _ = self._new_reset()
        raw = env._get_raw_obs()
        np.testing.assert_allclose(raw[env.OBS_TIME], [0.0, 1.0], atol=1e-14)
        self.assertEqual(float(raw[env.OBS_BESS_SOC][0]), 0.50)
        self.assertEqual(float(obs[env.OBS_BESS_SOC][0]), 0.50)
        self.assertEqual(env.bess.E_kWh, 125.0)

    def test_repeated_get_obs_is_idempotent_and_does_not_advance_components(self):
        env, obs_reset, _ = self._new_reset(seed=42)
        snapshot = {
            "t": env.t,
            "bess_soc": env.bess.soc,
            "zoe_soc": env.zoe_fleet.soc.copy(),
            "zoe_eac_rem": env.zoe_fleet.Eac_rem.copy(),
            "zoe_t": env.zoe_fleet.t,
            "truck5_state": env.truck5.state_at(0).copy(),
            "truck6_state": env.truck6.state_at(0).copy(),
            "truck5_t": env.truck5.t,
            "truck6_t": env.truck6.t,
            "office_t": env.office_sampler.t,
            "wh1_t": env.wh1_sampler.t,
            "wh2_t": env.wh2_sampler.t,
            "pv_t": (env.pv1.t, env.pv2.t, env.pv3.t),
            "pv_sampler_t": (env.pv1.sampler.t, env.pv2.sampler.t, env.pv3.sampler.t),
            "pf": (env.pf_office, env.pf_wh1, env.pf_wh2),
            "weights": (env.w_office.copy(), env.w_wh1.copy(), env.w_wh2.copy()),
        }
        raw_a = env._get_raw_obs()
        obs_a = env._get_obs()
        raw_b = env._get_raw_obs()
        obs_b = env._get_obs()

        np.testing.assert_array_equal(obs_reset, obs_a)
        np.testing.assert_array_equal(raw_a, raw_b)
        np.testing.assert_array_equal(obs_a, obs_b)
        self.assertEqual(env.t, snapshot["t"])
        self.assertEqual(env.bess.soc, snapshot["bess_soc"])
        np.testing.assert_array_equal(env.zoe_fleet.soc, snapshot["zoe_soc"])
        np.testing.assert_array_equal(env.zoe_fleet.Eac_rem, snapshot["zoe_eac_rem"])
        self.assertEqual(env.zoe_fleet.t, snapshot["zoe_t"])
        np.testing.assert_array_equal(env.truck5.state_at(0), snapshot["truck5_state"])
        np.testing.assert_array_equal(env.truck6.state_at(0), snapshot["truck6_state"])
        self.assertEqual((env.truck5.t, env.truck6.t), (snapshot["truck5_t"], snapshot["truck6_t"]))
        self.assertEqual((env.office_sampler.t, env.wh1_sampler.t, env.wh2_sampler.t), (0, 0, 0))
        self.assertEqual((env.pv1.t, env.pv2.t, env.pv3.t), snapshot["pv_t"])
        self.assertEqual((env.pv1.sampler.t, env.pv2.sampler.t, env.pv3.sampler.t), snapshot["pv_sampler_t"])
        self.assertEqual((env.pf_office, env.pf_wh1, env.pf_wh2), snapshot["pf"])
        for actual, expected in zip((env.w_office, env.w_wh1, env.w_wh2), snapshot["weights"]):
            np.testing.assert_array_equal(actual, expected)

    def test_same_seed_same_instance_and_different_instances_are_identical(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        obs_first, info_first = env.reset(seed=123)
        obs_second, info_second = env.reset(seed=123)
        np.testing.assert_array_equal(obs_first, obs_second)
        self.assertEqual(info_first["week_indices"], info_second["week_indices"])
        self.assertEqual(info_first["pf_office"], info_second["pf_office"])
        np.testing.assert_array_equal(info_first["w_office"], info_second["w_office"])

        env_a, obs_a, info_a = self._new_reset(seed=123)
        env_b, obs_b, info_b = self._new_reset(seed=123)
        np.testing.assert_array_equal(obs_a, obs_b)
        self.assertEqual(info_a["week_indices"], info_b["week_indices"])
        self.assertEqual(info_a["pf_wh2"], info_b["pf_wh2"])
        np.testing.assert_array_equal(info_a["w_wh2"], info_b["w_wh2"])
        self.assertEqual(env_a.component_seeds, env_b.component_seeds)

    def test_different_seed_changes_stochastic_state(self):
        _, obs_a, _ = self._new_reset(seed=123)
        _, obs_b, _ = self._new_reset(seed=456)
        self.assertFalse(np.array_equal(obs_a, obs_b))

    def test_pf_and_phase_weights_follow_episode_contract(self):
        env, _, _ = self._new_reset(seed=42)
        self.assertGreaterEqual(env.pf_office, 0.80)
        self.assertLessEqual(env.pf_office, 0.995)
        for pf in (env.pf_wh1, env.pf_wh2):
            self.assertGreaterEqual(pf, 0.57)
            self.assertLessEqual(pf, 0.88)
        for weights in (env.w_office, env.w_wh1, env.w_wh2):
            self.assertTrue(np.all(weights > 0.0))
            np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-14)
            self.assertTrue(np.all(weights >= env.PHASE_WEIGHT_RANGE[0]))
            self.assertTrue(np.all(weights <= env.PHASE_WEIGHT_RANGE[1]))

    def test_bus3_adds_zoes_to_phases_one_and_two(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.zoe_fleet.start_hour = 0
        _, _ = env.reset(seed=42)
        raw = env._get_raw_obs()
        p_office_total = env.office_sampler.value_at(0)
        q_office_total = env.compute_reactive_power(p_office_total, env.pf_office)
        p_office_abc, q_office_abc = env.split_three_phase(
            p_office_total, q_office_total, env.w_office
        )
        p_zoe = env.zoe_fleet.get_power_request(0)
        expected_p = p_office_abc + np.array([p_zoe[0], p_zoe[1], 0.0])
        np.testing.assert_allclose(raw[env.OBS_BUS3_LOAD.start : env.OBS_BUS3_LOAD.start + 3], expected_p)
        np.testing.assert_allclose(raw[env.OBS_BUS3_LOAD.start + 3 : env.OBS_BUS3_LOAD.stop], q_office_abc)
        self.assertTrue(np.all(p_zoe > 0.0))

    def test_forecasts_are_eleven_future_values_at_time_zero(self):
        env, _, _ = self._new_reset(seed=42)
        raw = env._get_raw_obs()
        expected = (
            (env.OBS_FORECAST_OFFICE, env.office_sampler.forecast_at(0, horizon=11)),
            (env.OBS_FORECAST_WH1, env.wh1_sampler.forecast_at(0, horizon=11)),
            (env.OBS_FORECAST_WH2, env.wh2_sampler.forecast_at(0, horizon=11)),
            (env.OBS_FORECAST_PV1, env.pv1.forecast_at(0, horizon=11)),
            (env.OBS_FORECAST_PV2, env.pv2.forecast_at(0, horizon=11)),
            (env.OBS_FORECAST_PV3, env.pv3.forecast_at(0, horizon=11)),
        )
        for obs_slice, forecast in expected:
            self.assertEqual(forecast.shape, (11,))
            np.testing.assert_allclose(raw[obs_slice], forecast)

    def test_raw_units_and_normalization_scales(self):
        env, obs, _ = self._new_reset(seed=42)
        raw = env._get_raw_obs()
        self.assertEqual(raw.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(raw)))
        self.assertTrue(np.all(raw[env.OBS_BUS3_LOAD] >= 0.0))
        self.assertTrue(np.all(raw[env.OBS_PV_AVAILABLE] >= 0.0))
        self.assertEqual(raw[env.OBS_BESS_SOC][0], 0.50)

        bus3_p_scale = np.array([40.74, 40.74, 18.74])
        q_office_scale = env.OFFICE_PMAX_KW * np.tan(np.arccos(0.80))
        wh1_q_scale = env.WH1_PMAX_KW * np.tan(np.arccos(0.57))
        wh2_q_scale = env.WH2_PMAX_KW * np.tan(np.arccos(0.57))
        np.testing.assert_allclose(obs[2:5], raw[2:5] / bus3_p_scale)
        np.testing.assert_allclose(obs[5:8], raw[5:8] / q_office_scale)
        np.testing.assert_allclose(obs[8:11], raw[8:11] / env.WH1_PMAX_KW)
        np.testing.assert_allclose(obs[11:14], raw[11:14] / wh1_q_scale)
        np.testing.assert_allclose(obs[14:17], raw[14:17] / env.WH2_PMAX_KW)
        np.testing.assert_allclose(obs[17:20], raw[17:20] / wh2_q_scale)
        np.testing.assert_allclose(obs[20:23], raw[20:23] / [74.25, 75.58, 7.75])
        np.testing.assert_allclose(obs[23], raw[23])

    def test_twenty_resets_are_finite_and_inside_observation_space(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        for seed in range(20):
            with self.subTest(seed=seed):
                obs, _ = env.reset(seed=seed)
                self.assertTrue(np.all(np.isfinite(obs)))
                self.assertTrue(env.observation_space.contains(obs))


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
