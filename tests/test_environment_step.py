from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import MicrorredIndustrial


class MicrorredIndustrialStepTests(unittest.TestCase):
    @staticmethod
    def _zero_action() -> np.ndarray:
        return np.zeros(12, dtype=np.float32)

    @staticmethod
    def _transactional_state(env: MicrorredIndustrial):
        return {
            "t": env.t,
            "bess_energy": env.bess.E_kWh,
            "truck5": (
                env.truck5.t,
                env.truck5.soc,
                env.truck5.P_grid_kw,
                env.truck5.Q_grid_kvar,
                env.truck5._action_applied_t,
            ),
            "truck6": (
                env.truck6.t,
                env.truck6.soc,
                env.truck6.P_grid_kw,
                env.truck6.Q_grid_kvar,
                env.truck6._action_applied_t,
            ),
            "zoe": (
                env.zoe_fleet.t,
                env.zoe_fleet.soc.copy(),
                env.zoe_fleet.Eac_rem.copy(),
                env.zoe_fleet.connected.copy(),
            ),
        }

    def test_step_requires_reset_and_reuses_decoder_validation(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step(self._zero_action())

        env.reset(seed=5)
        with self.assertRaisesRegex(ValueError, "shape"):
            env.step(np.zeros(11, dtype=np.float32))
        self.assertEqual(env.t, 0)

    def test_first_step_uses_hour_zero_and_commits_resolved_soc_once(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.reset(seed=42)
        action = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.25, -0.2, 0.3, 0.4, -0.5],
            dtype=np.float32,
        )
        decoded = env._decode_action(action, t=0)
        expected_bess = env.bess.resolve_action(
            decoded["bess"]["u_internal_kw"], env.dt_h, decoded["bess"]["q_frac"]
        )
        expected_truck5 = env.truck5.resolve_action_at(
            0, decoded["truck5"]["aP_internal"], decoded["truck5"]["aQ_internal"]
        )
        expected_truck6 = env.truck6.resolve_action_at(
            0, decoded["truck6"]["aP_internal"], decoded["truck6"]["aQ_internal"]
        )

        next_obs, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(env.t, 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["t"], 0)
        self.assertEqual(info["next_t"], 1)
        self.assertEqual(info["price_raw"], env.price_sampler.value_at(0))
        self.assertEqual(info["p_grid_kw"], info["pcc_p_kw"])
        self.assertEqual(reward, info["reward"])
        self.assertAlmostEqual(reward, -info["operating_cost_usd"] - info["pf_penalty_usd"])
        self.assertAlmostEqual(env.bess.soc, expected_bess.soc_next)
        self.assertAlmostEqual(env.truck5.soc, expected_truck5.soc_next)
        self.assertAlmostEqual(env.truck6.soc, expected_truck6.soc_next)
        self.assertTrue(info["electrical_converged"])
        self.assertEqual(np.asarray(info["pf_phase"]).shape, (3,))
        self.assertIn("voltages_pu", info)
        self.assertIn("transformer_loading_pct", info)
        self.assertIn("controls", info)
        self.assertEqual(next_obs.shape, (114,))
        self.assertEqual(next_obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(next_obs))
        np.testing.assert_allclose(next_obs, env._get_obs())

    def test_zoe_request_is_sent_to_snapshot_and_integrated_once(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.reset(seed=11)
        for _ in range(9):
            env.step(self._zero_action())

        self.assertEqual(env.t, 9)
        hour_state = env._hour_state_cache
        request = env.zoe_fleet.get_power_request(9)
        soc_before = env.zoe_fleet.soc.copy()
        expected_soc = np.minimum(
            env.zoe_fleet.soc_target,
            soc_before + env.zoe_fleet.eta * request / env.zoe_fleet.C,
        )
        original_solver = env._solve_electrical_snapshot
        with patch.object(env, "_solve_electrical_snapshot", wraps=original_solver) as solve:
            _, _, terminated, truncated, info = env.step(self._zero_action())

        snapshot_inputs = solve.call_args.args[0]
        np.testing.assert_allclose(snapshot_inputs.p_bus3_abc_kw, hour_state["p_bus3_abc"])
        np.testing.assert_allclose(info["controls"]["zoe"]["power_kw"], request)
        np.testing.assert_allclose(env.zoe_fleet.soc, expected_soc)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_off_by_one_uses_hour_ten_price_and_returns_hour_eleven_observation(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.reset(seed=17)
        for _ in range(10):
            env.step(self._zero_action())

        self.assertEqual(env.t, 10)
        _, _, terminated, truncated, info = env.step(self._zero_action())

        self.assertEqual(info["t"], 10)
        self.assertEqual(info["next_t"], 11)
        self.assertEqual(info["price_raw"], env.price_sampler.value_at(10))
        self.assertEqual(env.t, 11)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_nonconvergence_keeps_dynamic_state_uncommitted(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.reset(seed=23)
        before = self._transactional_state(env)

        with patch.object(env, "_solve_electrical_snapshot", return_value=SimpleNamespace(converged=False)):
            with self.assertRaisesRegex(RuntimeError, "did not converge.*t=0"):
                env.step(self._zero_action())

        after = self._transactional_state(env)
        self.assertEqual(before["t"], after["t"])
        self.assertEqual(before["bess_energy"], after["bess_energy"])
        self.assertEqual(before["truck5"], after["truck5"])
        self.assertEqual(before["truck6"], after["truck6"])
        np.testing.assert_allclose(before["zoe"][1], after["zoe"][1])
        np.testing.assert_allclose(before["zoe"][2], after["zoe"][2])
        np.testing.assert_array_equal(before["zoe"][3], after["zoe"][3])
        self.assertEqual(before["zoe"][0], after["zoe"][0])

    def test_solver_exception_keeps_dynamic_state_uncommitted(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.reset(seed=29)
        before = self._transactional_state(env)

        with patch.object(env, "_solve_electrical_snapshot", side_effect=RuntimeError("synthetic solver error")):
            with self.assertRaisesRegex(RuntimeError, "synthetic solver error"):
                env.step(self._zero_action())

        after = self._transactional_state(env)
        self.assertEqual(before["t"], after["t"])
        self.assertEqual(before["bess_energy"], after["bess_energy"])
        self.assertEqual(before["truck5"], after["truck5"])
        self.assertEqual(before["truck6"], after["truck6"])
        np.testing.assert_allclose(before["zoe"][1], after["zoe"][1])
        np.testing.assert_allclose(before["zoe"][2], after["zoe"][2])
        np.testing.assert_array_equal(before["zoe"][3], after["zoe"][3])
        self.assertEqual(before["zoe"][0], after["zoe"][0])

    def test_terminal_step_uses_sentinel_and_requires_reset(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.reset(seed=31)
        for step_index in range(env.horizon):
            obs, _, terminated, truncated, info = env.step(self._zero_action())
            if step_index < env.horizon - 1:
                self.assertFalse(terminated)
                self.assertFalse(truncated)
            else:
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertTrue(info["terminal_observation_is_sentinel"])

        self.assertEqual(env.t, 168)
        self.assertEqual(obs.shape, (114,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        np.testing.assert_array_equal(obs, np.zeros(114, dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "Episode has terminated"):
            env.step(self._zero_action())

        env.reset(seed=31)
        self.assertEqual(env.t, 0)
        env.step(self._zero_action())
        self.assertEqual(env.t, 1)


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
