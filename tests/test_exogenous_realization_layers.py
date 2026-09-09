"""Contratos base -> realización -> forecast de cargas y generación PV."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import MicrorredIndustrial
from env.PV_plant import WeeklyPVSampler
from env.oficinas import OfficeLoadSampler


DATA_DIR = CODE_DIR / "data" / "train"


class ExogenousRealizationLayerTests(unittest.TestCase):
    LOADS = (
        ("office", "office_weeks_pool.npz", 18.74),
        ("wh1", "wh1_weeks_pool.npz", 65.66),
        ("wh2", "wh2_weeks_pool.npz", 76.12),
    )
    PV_PMAX = (74.25, 75.58, 7.75)

    @staticmethod
    def _load_sampler(filename: str, pmax_kw: float) -> OfficeLoadSampler:
        return OfficeLoadSampler(DATA_DIR / filename, pmax_target_kw=pmax_kw)

    def test_loads_follow_base_realization_forecast_with_fixed_base(self):
        for name, filename, pmax_kw in self.LOADS:
            with self.subTest(load=name):
                sampler_a = self._load_sampler(filename, pmax_kw)
                sampler_b = self._load_sampler(filename, pmax_kw)
                sampler_c = self._load_sampler(filename, pmax_kw)
                real_a, forecast_a = sampler_a.reset(seed=123, week_idx=0)
                real_b, forecast_b = sampler_b.reset(seed=123, week_idx=0)
                real_c, forecast_c = sampler_c.reset(seed=456, week_idx=0)

                np.testing.assert_array_equal(sampler_a.P_base_profile, sampler_b.P_base_profile)
                np.testing.assert_array_equal(sampler_a.P_base_profile, sampler_c.P_base_profile)
                np.testing.assert_array_equal(real_a, real_b)
                np.testing.assert_array_equal(forecast_a, forecast_b)
                self.assertFalse(np.array_equal(real_a, real_c))
                self.assertFalse(np.array_equal(forecast_a, forecast_c))
                self.assertTrue(np.all(real_a >= 0.0))
                np.testing.assert_allclose(
                    real_a,
                    np.maximum(
                        sampler_a.P_base_profile + sampler_a.realization_noise,
                        0.0,
                    ),
                )
                np.testing.assert_allclose(
                    sampler_a.P_forecast_ext,
                    sampler_a.P_real_ext + sampler_a.forecast_noise_ext,
                )

                t = next(
                    hour
                    for hour in range(sampler_a.horizon_hours - sampler_a.forecast_h)
                    if np.any(np.abs(sampler_a.realization_noise[hour + 1 : hour + 12]) > 1e-12)
                )
                forecast = sampler_a.forecast_at(t)
                np.testing.assert_allclose(
                    forecast - sampler_a.P_real_ext[t + 1 : t + 12],
                    sampler_a.forecast_noise_ext[t + 1 : t + 12],
                )
                self.assertFalse(
                    np.allclose(
                        forecast,
                        sampler_a.P_base_ext[t + 1 : t + 12]
                        + sampler_a.forecast_noise_ext[t + 1 : t + 12],
                    )
                )
                edge_forecast = sampler_a.forecast_at(167, horizon=11)
                np.testing.assert_allclose(
                    edge_forecast - sampler_a.P_real_ext[168:179],
                    sampler_a.forecast_noise_ext[168:179],
                )

    def test_load_realization_statistics_and_pmax_is_not_a_hard_cap(self):
        for name, filename, pmax_kw in self.LOADS:
            with self.subTest(load=name):
                sampler = self._load_sampler(filename, pmax_kw)
                sampler.reset(seed=0, week_idx=0)
                base = sampler.P_base_profile.copy()
                hour = int(np.argmax(base))
                samples = np.asarray(
                    [sampler.reset(seed=seed, week_idx=0)[0][hour] for seed in range(1024)],
                    dtype=float,
                )
                self.assertAlmostEqual(float(samples.mean()), float(base[hour]), delta=0.015 * base[hour])
                self.assertAlmostEqual(float(samples.std(ddof=1)), 0.10 * base[hour], delta=0.012 * base[hour])
                self.assertTrue(np.all(samples >= 0.0))
                self.assertTrue(np.any(samples > pmax_kw))

    def test_environment_uses_real_load_pq_and_unbounded_current_load_box(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        obs, _ = env.reset(seed=42)
        hour = env._collect_hour_state(0)
        expected = (
            (env.office_sampler, env.pf_office, env.w_office, hour["p_bus3_abc"] - np.r_[hour["p_zoe"], 0.0]),
            (env.wh1_sampler, env.pf_wh1, env.w_wh1, hour["p_wh1_abc"]),
            (env.wh2_sampler, env.pf_wh2, env.w_wh2, hour["p_wh2_abc"]),
        )
        for sampler, pf, weights, p_abc in expected:
            p_real = sampler.value_at(0)
            q_real = env.compute_reactive_power(p_real, pf)
            expected_p, expected_q = env.split_three_phase(p_real, q_real, weights)
            np.testing.assert_allclose(p_abc, expected_p)
            if sampler is env.office_sampler:
                np.testing.assert_allclose(hour["q_bus3_abc"], expected_q)
            elif sampler is env.wh1_sampler:
                np.testing.assert_allclose(hour["q_wh1_abc"], expected_q)
            else:
                np.testing.assert_allclose(hour["q_wh2_abc"], expected_q)

        self.assertEqual(obs.shape, (114,))
        for load_slice in (env.OBS_BUS3_LOAD, env.OBS_WH1_LOAD, env.OBS_WH2_LOAD):
            self.assertTrue(np.all(np.isposinf(env.observation_space.high[load_slice])))

    def test_pv_pipeline_shares_common_real_weather_and_forecasts_final_power(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        _, info = env.reset(seed=42)
        plants = (env.pv1, env.pv2, env.pv3)
        samplers = tuple(plant.sampler for plant in plants)

        self.assertTrue(all(sampler.week_idx == info["pv_week_idx"] for sampler in samplers))
        for sampler in samplers[1:]:
            np.testing.assert_array_equal(samplers[0].common_base_weather, sampler.common_base_weather)
            np.testing.assert_array_equal(samplers[0].common_real_weather, sampler.common_real_weather)
        np.testing.assert_allclose(
            samplers[0].common_real_weather,
            np.maximum(
                samplers[0].common_base_weather + samplers[0].common_realization_noise,
                0.0,
            ),
        )

        night = samplers[0].common_base_weather == 0.0
        np.testing.assert_array_equal(samplers[0].common_real_weather[night], 0.0)
        normalized_real = []
        for plant, sampler, pmax_kw in zip(plants, samplers, self.PV_PMAX):
            expected_real = np.clip(
                sampler.P_common_real_week * (1.0 + sampler.spatial_epsilon_week),
                0.0,
                pmax_kw,
            )
            np.testing.assert_allclose(plant.P_avail_week, expected_real)
            self.assertTrue(np.all((0.0 <= plant.P_avail_week) & (plant.P_avail_week <= pmax_kw)))
            np.testing.assert_array_equal(plant.P_avail_week[night], 0.0)
            np.testing.assert_allclose(
                sampler.P_forecast_ext,
                sampler.P_real_ext + sampler.forecast_noise_ext,
            )
            forecast = plant.forecast_at(0)
            np.testing.assert_allclose(
                forecast - sampler.P_real_ext[1:12],
                sampler.forecast_noise_ext[1:12],
            )
            edge_forecast = plant.forecast_at(167, horizon=11)
            np.testing.assert_allclose(
                edge_forecast - sampler.P_real_ext[168:179],
                sampler.forecast_noise_ext[168:179],
            )
            base_ext = np.r_[sampler.P_base_week, sampler.P_base_week[:11]]
            self.assertFalse(np.allclose(forecast, base_ext[1:12] + sampler.forecast_noise_ext[1:12]))
            normalized_real.append(plant.P_avail_week / pmax_kw)

        sunlight = ~night
        for first, second in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(np.allclose(normalized_real[first][sunlight], normalized_real[second][sunlight]))
            self.assertGreater(np.corrcoef(normalized_real[first][sunlight], normalized_real[second][sunlight])[0, 1], 0.95)

    def test_pv_common_weather_statistics_conditioned_on_fixed_base(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.reset(seed=42)
        base = env.pv1.sampler.common_base_weather.copy()
        hour = int(np.argmax(base))
        samples = np.asarray(
            [
                WeeklyPVSampler.sample_common_real_weather(base, seed=seed)[0][hour]
                for seed in range(1024)
            ],
            dtype=float,
        )
        self.assertGreater(base[hour], 0.0)
        self.assertAlmostEqual(float(samples.mean()), float(base[hour]), delta=0.015 * base[hour])
        self.assertAlmostEqual(float(samples.std(ddof=1)), 0.10 * base[hour], delta=0.012 * base[hour])


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
