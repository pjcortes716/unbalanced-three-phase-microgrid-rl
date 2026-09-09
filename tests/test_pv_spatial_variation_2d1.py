from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import MicrorredIndustrial


class PVSpatialVariation2D1Tests(unittest.TestCase):
    PMAX = (74.25, 75.58, 7.75)

    def _reset(self, seed=42):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        _, info = env.reset(seed=seed)
        return env, info

    @staticmethod
    def _plants(env):
        return (env.pv1, env.pv2, env.pv3)

    def test_common_week_and_common_base_profile_are_preserved(self):
        env, info = self._reset()
        plants = self._plants(env)
        self.assertEqual(info["pv_week_idx"], plants[0].sampler.week_idx)
        self.assertTrue(all(p.sampler.week_idx == info["pv_week_idx"] for p in plants))
        base_normalized = [p.sampler.P_base_week / pmax for p, pmax in zip(plants, self.PMAX)]
        np.testing.assert_allclose(base_normalized[0], base_normalized[1])
        np.testing.assert_allclose(base_normalized[0], base_normalized[2])

    def test_real_profiles_are_not_clones_but_remain_highly_correlated(self):
        env, _ = self._reset()
        plants = self._plants(env)
        base = plants[0].sampler.P_base_week
        sunlight = base > 0.0
        normalized = [p.P_avail_week / pmax for p, pmax in zip(plants, self.PMAX)]

        for first, second in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(np.allclose(normalized[first][sunlight], normalized[second][sunlight]))
            self.assertGreater(np.corrcoef(normalized[first][sunlight], normalized[second][sunlight])[0, 1], 0.95)

    def test_night_remains_zero_and_profiles_respect_finite_pmax_limits(self):
        env, _ = self._reset()
        plants = self._plants(env)
        night = plants[0].sampler.P_base_week == 0.0
        for plant, pmax in zip(plants, self.PMAX):
            np.testing.assert_array_equal(plant.P_avail_week[night], np.zeros(np.count_nonzero(night)))
            self.assertTrue(np.all(np.isfinite(plant.P_avail_week)))
            self.assertGreaterEqual(float(np.min(plant.P_avail_week)), 0.0)
            self.assertLessEqual(float(np.max(plant.P_avail_week)), pmax)

    def test_perturbed_profiles_are_reproducible_and_change_with_seed(self):
        env_a, _ = self._reset(seed=42)
        env_b, _ = self._reset(seed=42)
        env_c, _ = self._reset(seed=456)
        for plant_a, plant_b in zip(self._plants(env_a), self._plants(env_b)):
            np.testing.assert_array_equal(plant_a.P_avail_week, plant_b.P_avail_week)
            np.testing.assert_array_equal(
                plant_a.sampler.spatial_epsilon_week,
                plant_b.sampler.spatial_epsilon_week,
            )
        self.assertTrue(
            any(
                not np.array_equal(plant_a.P_avail_week, plant_c.P_avail_week)
                for plant_a, plant_c in zip(self._plants(env_a), self._plants(env_c))
            )
        )

    def test_available_at_is_idempotent_and_does_not_consume_rng(self):
        env, _ = self._reset()
        plant = env.pv1
        sampler = plant.sampler
        t = next(hour for hour, value in enumerate(plant.P_avail_week) if value > 0.0)
        snapshot = {
            "plant_t": plant.t,
            "sampler_t": sampler.t,
            "real": sampler.P_real_ext.copy(),
            "epsilon": sampler.spatial_epsilon_week.copy(),
            "rng": deepcopy(sampler.rng.bit_generator.state),
        }
        first = plant.available_at(t)
        second = plant.available_at(t)
        self.assertEqual(first, second)
        self.assertEqual(plant.t, snapshot["plant_t"])
        self.assertEqual(sampler.t, snapshot["sampler_t"])
        np.testing.assert_array_equal(sampler.P_real_ext, snapshot["real"])
        np.testing.assert_array_equal(sampler.spatial_epsilon_week, snapshot["epsilon"])
        self.assertEqual(sampler.rng.bit_generator.state, snapshot["rng"])

    def test_forecast_uses_final_real_profile_without_clipping_and_has_eleven_future_values(self):
        env, _ = self._reset()
        for plant in self._plants(env):
            sampler = plant.sampler
            forecast = plant.forecast_at(0)
            self.assertEqual(forecast.shape, (11,))
            expected = sampler.P_real_ext + sampler.forecast_noise_ext
            np.testing.assert_allclose(sampler.P_forecast_ext, expected)
            np.testing.assert_allclose(forecast, expected[1:12])
            np.testing.assert_allclose(
                forecast - sampler.P_real_ext[1:12],
                sampler.forecast_noise_ext[1:12],
            )


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
