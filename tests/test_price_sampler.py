"""Contrato temporal y de observación para el precio exógeno."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import ElectricalSnapshotInputs, MicrorredIndustrial
from env.price_sampler import PRICE_PROFILE_24H, PRICE_SCALE, PriceSampler


class PriceSamplerTests(unittest.TestCase):
    def test_01_daily_profile_is_the_base_mean_not_the_realization(self):
        sampler = PriceSampler(seed=1)
        real_week, forecast_ext = sampler.reset(seed=7)

        expected = np.tile(PRICE_PROFILE_24H, 7)
        np.testing.assert_array_equal(sampler.base_price, expected)
        self.assertFalse(np.array_equal(real_week, expected))
        self.assertEqual(forecast_ext.shape, (179,))
        self.assertEqual(sampler.value_at(0), real_week[0])
        self.assertEqual(sampler.value_at(8), real_week[8])
        self.assertEqual(sampler.value_at(24), real_week[24])
        self.assertEqual(sampler.value_at(32), real_week[32])
        np.testing.assert_array_equal(sampler.price_real_ext[168:179], real_week[:11])

    def test_02_forecast_is_future_only_idempotent_and_cyclic_at_episode_edge(self):
        sampler = PriceSampler(seed=1)
        sampler.reset(seed=123)
        real_before = sampler.price_real_ext.copy()
        forecast_before = sampler.price_forecast_ext.copy()

        first = sampler.forecast_at(0, horizon=11)
        second = sampler.forecast_at(0, horizon=11)
        edge = sampler.forecast_at(167, horizon=11)

        self.assertEqual(first.shape, (11,))
        self.assertTrue(np.all(np.isfinite(first)))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, sampler.price_forecast_ext[1:12])
        np.testing.assert_array_equal(edge, sampler.price_forecast_ext[168:179])
        np.testing.assert_array_equal(sampler.price_real_ext, real_before)
        np.testing.assert_array_equal(sampler.price_forecast_ext, forecast_before)
        self.assertFalse(hasattr(sampler, "t"))

    def test_03_same_seed_reproduces_realization_and_forecast(self):
        sampler_a = PriceSampler()
        sampler_b = PriceSampler()
        sampler_c = PriceSampler()
        real_a, forecast_a = sampler_a.reset(seed=42)
        real_b, forecast_b = sampler_b.reset(seed=42)
        real_c, forecast_c = sampler_c.reset(seed=43)

        np.testing.assert_array_equal(real_a, real_b)
        np.testing.assert_array_equal(forecast_a, forecast_b)
        self.assertFalse(np.array_equal(real_a, real_c))
        self.assertFalse(np.array_equal(forecast_a, forecast_c))

    def test_04_realization_statistics_match_the_base_mean_and_std(self):
        hour = 8
        base = PRICE_PROFILE_24H[hour]
        samples = np.asarray(
            [PriceSampler().reset(seed=seed)[0][hour] for seed in range(2048)],
            dtype=float,
        )

        self.assertAlmostEqual(float(samples.mean()), float(base), delta=0.012 * base)
        self.assertAlmostEqual(float(samples.std(ddof=1)), 0.10 * base, delta=0.010 * base)

    def test_05_forecast_error_is_relative_to_the_real_future_price(self):
        sampler = PriceSampler()
        sampler.reset(seed=123)

        future_real = sampler.price_real_ext[1:12]
        future_base = sampler.base_price_ext[1:12]
        forecast = sampler.forecast_at(0, horizon=11)
        forecast_noise = sampler.forecast_noise_ext[1:12]

        np.testing.assert_allclose(sampler.price_real, sampler.base_price + sampler.realization_noise)
        np.testing.assert_allclose(sampler.price_forecast_ext, sampler.price_real_ext + sampler.forecast_noise_ext)
        np.testing.assert_allclose(forecast - future_real, forecast_noise)
        self.assertFalse(np.allclose(forecast, future_base + forecast_noise))

    def test_06_environment_appends_real_price_after_unchanged_legacy_observation(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        obs, info = env.reset(seed=42)
        raw = env._get_raw_obs()
        hour = env._collect_hour_state(0)

        legacy_raw = np.concatenate(
            (
                [0.0, 1.0],
                hour["p_bus3_abc"], hour["q_bus3_abc"],
                hour["p_wh1_abc"], hour["q_wh1_abc"],
                hour["p_wh2_abc"], hour["q_wh2_abc"],
                hour["pv_available"],
                [env.bess.soc],
                hour["forecasts"]["office"],
                hour["forecasts"]["wh1"],
                hour["forecasts"]["wh2"],
                hour["forecasts"]["pv1"],
                hour["forecasts"]["pv2"],
                hour["forecasts"]["pv3"],
                hour["truck5_state"],
                hour["truck6_state"],
                hour["zoe_soc"],
            )
        )
        self.assertEqual(legacy_raw.shape, (102,))
        np.testing.assert_allclose(raw[:102], legacy_raw)
        self.assertEqual(obs.shape, (114,))
        self.assertEqual(env.observation_space.shape, (114,))
        self.assertEqual(raw[env.OBS_CURRENT_PRICE][0], env.price_sampler.value_at(0))
        self.assertNotEqual(raw[env.OBS_CURRENT_PRICE][0], PRICE_PROFILE_24H[0])
        np.testing.assert_allclose(
            raw[env.OBS_FORECAST_PRICE], env.price_sampler.forecast_at(0, horizon=11)
        )
        self.assertAlmostEqual(
            obs[102], env.price_sampler.value_at(0) / PRICE_SCALE, places=7
        )
        np.testing.assert_allclose(
            obs[env.OBS_FORECAST_PRICE],
            env.price_sampler.forecast_at(0, horizon=11) / PRICE_SCALE,
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertEqual(info["current_price"], env.price_sampler.value_at(0))

    def test_07_environment_real_price_and_forecast_are_seed_dependent(self):
        env_a = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env_b = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env_c = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        obs_a, _ = env_a.reset(seed=123)
        obs_b, _ = env_b.reset(seed=123)
        obs_c, _ = env_c.reset(seed=456)

        self.assertEqual(env_a.price_sampler.value_at(0), env_b.price_sampler.value_at(0))
        self.assertNotEqual(env_a.price_sampler.value_at(0), env_c.price_sampler.value_at(0))
        np.testing.assert_array_equal(
            env_a.price_sampler.forecast_at(0), env_b.price_sampler.forecast_at(0)
        )
        self.assertFalse(
            np.array_equal(
                env_a.price_sampler.price_real,
                env_c.price_sampler.price_real,
            )
        )
        self.assertFalse(np.array_equal(obs_a[103:114], obs_c[103:114]))

    def test_08_price_queries_do_not_change_electrical_snapshot(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.reset(seed=42)
        hour = env._collect_hour_state(env.t)
        inputs = ElectricalSnapshotInputs(
            p_bus3_abc_kw=hour["p_bus3_abc"],
            q_bus3_abc_kvar=hour["q_bus3_abc"],
            p_wh1_abc_kw=hour["p_wh1_abc"],
            q_wh1_abc_kvar=hour["q_wh1_abc"],
            p_wh2_abc_kw=hour["p_wh2_abc"],
            q_wh2_abc_kvar=hour["q_wh2_abc"],
            pv1_p_inj_kw=hour["pv_available"][0],
            pv2_p_inj_kw=hour["pv_available"][1],
            pv3_p_inj_kw=hour["pv_available"][2],
        )
        before = env._solve_electrical_snapshot(inputs)
        time_before = env.t
        price_before = env.price_sampler.value_at(0)
        forecast_before = env.price_sampler.forecast_at(0)
        _ = env.price_sampler.value_at(24)
        _ = env.price_sampler.forecast_at(167)
        after = env._solve_electrical_snapshot(inputs)

        self.assertEqual(env.t, time_before)
        self.assertEqual(env.price_sampler.value_at(0), price_before)
        np.testing.assert_array_equal(env.price_sampler.forecast_at(0), forecast_before)
        np.testing.assert_allclose(before.voltages_vpu["bus6"], after.voltages_vpu["bus6"])
        np.testing.assert_allclose(before.pcc_p_kw_phase, after.pcc_p_kw_phase)
        np.testing.assert_allclose(before.pcc_q_kvar_phase, after.pcc_q_kvar_phase)
        self.assertAlmostEqual(before.max_voltage_unbalance_pct, after.max_voltage_unbalance_pct)
        self.assertAlmostEqual(before.pcc_pf_total, after.pcc_pf_total)


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
