"""Pruebas unitarias de la reward nominal, aislada de ``step()``."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import (
    ElectricalSnapshotInputs,
    MicrorredIndustrial,
    PF_MIN,
    compute_reward,
)


class RewardTests(unittest.TestCase):
    @staticmethod
    def _compute(
        *,
        p_grid_kw: float = 0.0,
        price_raw: float = 10.0,
        pf_phase=(0.95, 0.96, 0.97),
        q_pcc_kvar_phase=(0.0, 0.0, 0.0),
        dt_h: float = 1.0,
        lambda_pf: float = 1.0,
    ):
        return compute_reward(
            p_grid_kw=p_grid_kw,
            price_raw=price_raw,
            pf_phase=np.asarray(pf_phase, dtype=float),
            q_pcc_kvar_phase=np.asarray(q_pcc_kvar_phase, dtype=float),
            dt_h=dt_h,
            lambda_pf=lambda_pf,
        )

    def test_01_import_without_pf_penalty_uses_cents_to_usd(self):
        result = self._compute(
            p_grid_kw=100.0,
            price_raw=10.0,
            pf_phase=(0.95, 0.96, 0.97),
            q_pcc_kvar_phase=(15.0, -30.0, 45.0),
        )

        self.assertEqual(result.price_raw, 10.0)
        self.assertEqual(result.price_usd_per_kwh, 0.10)
        self.assertEqual(result.operating_cost_usd, 10.0)
        self.assertEqual(result.pf_penalty_usd, 0.0)
        self.assertEqual(result.reward, -10.0)
        np.testing.assert_array_equal(result.pf_violation_phase, np.zeros(3))

    def test_02_export_is_revenue(self):
        result = self._compute(
            p_grid_kw=-100.0,
            price_raw=10.0,
            pf_phase=(0.90, 0.95, 1.0),
            q_pcc_kvar_phase=(0.0, 0.0, 0.0),
        )

        self.assertEqual(result.operating_cost_usd, -10.0)
        self.assertEqual(result.pf_penalty_usd, 0.0)
        self.assertEqual(result.reward, 10.0)

    def test_03_pf_exactly_at_threshold_has_no_penalty(self):
        result = self._compute(
            pf_phase=(PF_MIN, PF_MIN, PF_MIN),
            q_pcc_kvar_phase=(100.0, -200.0, 300.0),
            lambda_pf=50.0,
        )

        np.testing.assert_array_equal(result.pf_violation_phase, np.zeros(3))
        self.assertEqual(result.pf_penalty_usd, 0.0)

    def test_04_one_violated_phase_matches_manual_penalty(self):
        result = self._compute(
            pf_phase=(0.95, 0.85, 0.92),
            q_pcc_kvar_phase=(10.0, 20.0, 30.0),
            lambda_pf=2.0,
        )

        self.assertAlmostEqual(result.pf_violation_phase[1], 0.05)
        self.assertAlmostEqual(result.pf_penalty_usd, 2.0)
        self.assertAlmostEqual(result.reward, -2.0)

    def test_05_multiple_violated_phases_match_manual_penalty(self):
        result = self._compute(
            pf_phase=(0.80, 0.91, 0.75),
            q_pcc_kvar_phase=(-10.0, 50.0, -4.0),
            dt_h=2.0,
            lambda_pf=3.0,
        )

        # 3 USD/(kVAr h) * 2 h * ((0.10 * 10 kVAr) + (0.15 * 4 kVAr)) = 9.6 USD.
        self.assertAlmostEqual(result.pf_penalty_usd, 9.6)
        self.assertAlmostEqual(result.reward, -9.6)
        np.testing.assert_allclose(result.pf_violation_phase, (0.10, 0.0, 0.15))

    def test_06_q_sign_does_not_change_the_penalty(self):
        common = {
            "pf_phase": (0.80, 0.85, 0.92),
            "lambda_pf": 1.7,
        }
        first = self._compute(q_pcc_kvar_phase=(20.0, -20.0, 20.0), **common)
        second = self._compute(q_pcc_kvar_phase=(-20.0, 20.0, -20.0), **common)

        self.assertEqual(first.pf_penalty_usd, second.pf_penalty_usd)
        self.assertEqual(first.reward, second.reward)

    def test_07_lambda_zero_one_and_two_scale_only_pf_penalty(self):
        kwargs = {
            "pf_phase": (0.80, 0.95, 0.90),
            "q_pcc_kvar_phase": (10.0, 20.0, 30.0),
        }
        zero = self._compute(lambda_pf=0.0, **kwargs)
        one = self._compute(lambda_pf=1.0, **kwargs)
        two = self._compute(lambda_pf=2.0, **kwargs)

        self.assertEqual(zero.pf_penalty_usd, 0.0)
        self.assertAlmostEqual(one.pf_penalty_usd, 1.0)
        self.assertAlmostEqual(two.pf_penalty_usd, 2.0)
        self.assertAlmostEqual(two.pf_penalty_usd, 2.0 * one.pf_penalty_usd)

    def test_08_rejects_invalid_inputs_and_negative_environment_lambda(self):
        valid = {
            "p_grid_kw": 0.0,
            "price_raw": 10.0,
            "pf_phase": np.array([0.95, 0.96, 0.97]),
            "q_pcc_kvar_phase": np.zeros(3),
            "dt_h": 1.0,
            "lambda_pf": 1.0,
        }
        invalid_cases = (
            {"pf_phase": np.array([0.95, 0.96])},
            {"q_pcc_kvar_phase": np.zeros((3, 1))},
            {"pf_phase": np.array([0.95, np.nan, 0.97])},
            {"q_pcc_kvar_phase": np.array([0.0, np.inf, 0.0])},
            {"p_grid_kw": np.inf},
            {"price_raw": np.nan},
            {"dt_h": 0.0},
            {"lambda_pf": -0.01},
            {"pf_phase": np.array([-0.01, 0.95, 0.97])},
            {"pf_phase": np.array([1.01, 0.95, 0.97])},
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    compute_reward(**(valid | overrides))

        with self.assertRaises(ValueError):
            MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, lambda_pf=-0.01)

    def test_09_environment_delegator_is_pure_and_uses_its_configured_lambda(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, seed=4, lambda_pf=2.0)
        env.reset(seed=4)
        state_before = (
            env.t,
            env.bess.soc,
            env.truck5.soc,
            env.truck6.soc,
            env.zoe_fleet.soc.copy(),
            env.price_sampler.value_at(env.t),
        )

        result = env._compute_reward(
            p_grid_kw=0.0,
            price_raw=10.0,
            pf_phase=np.array([0.85, 0.95, 0.95]),
            q_pcc_kvar_phase=np.array([20.0, 0.0, 0.0]),
        )

        self.assertEqual(env.lambda_pf, 2.0)
        self.assertAlmostEqual(result.pf_penalty_usd, 2.0)
        self.assertEqual(env.t, state_before[0])
        self.assertEqual(env.bess.soc, state_before[1])
        self.assertEqual(env.truck5.soc, state_before[2])
        self.assertEqual(env.truck6.soc, state_before[3])
        np.testing.assert_array_equal(env.zoe_fleet.soc, state_before[4])
        self.assertEqual(env.price_sampler.value_at(env.t), state_before[5])

    def test_10_real_seed_42_snapshot_feeds_reward_without_changing_observation(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, lambda_pf=1.0)
        observation, _ = env.reset(seed=42)
        snapshot = env._solve_electrical_snapshot(
            ElectricalSnapshotInputs(
                p_bus3_abc_kw=env.P_office_abc,
                q_bus3_abc_kvar=env.Q_office_abc,
                p_wh1_abc_kw=env.P_wh1_abc,
                q_wh1_abc_kvar=env.Q_wh1_abc,
                p_wh2_abc_kw=env.P_wh2_abc,
                q_wh2_abc_kvar=env.Q_wh2_abc,
            )
        )

        self.assertTrue(snapshot.converged)
        self.assertIsNotNone(snapshot.pcc)
        self.assertIsNotNone(snapshot.pcc_pf_phase)
        self.assertIsNotNone(snapshot.pcc_q_kvar_phase)
        result = env._compute_reward(
            p_grid_kw=snapshot.pcc["P_total_kW"],
            price_raw=env.price_sampler.value_at(env.t),
            pf_phase=snapshot.pcc_pf_phase,
            q_pcc_kvar_phase=snapshot.pcc_q_kvar_phase,
        )

        self.assertAlmostEqual(result.reward, -result.operating_cost_usd - result.pf_penalty_usd)
        self.assertEqual(result.price_raw, env.current_price)
        self.assertEqual(observation.shape, (114,))
        self.assertEqual(env._get_obs().shape, (114,))


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
