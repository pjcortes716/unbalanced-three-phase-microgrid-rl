from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import MicrorredIndustrial


class LoadTelemetryTests(unittest.TestCase):
    LOAD_NAMES = ("office", "zoe", "bus3", "wh1", "wh2", "total")

    @staticmethod
    def _zero_action() -> np.ndarray:
        return np.zeros(12, dtype=np.float32)

    @staticmethod
    def _assert_value_equal(test_case: unittest.TestCase, actual, expected) -> None:
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            test_case.assertEqual(set(actual), set(expected))
            for key in expected:
                LoadTelemetryTests._assert_value_equal(test_case, actual[key], expected[key])
        else:
            test_case.assertEqual(actual, expected)

    def _advance_to(self, env: MicrorredIndustrial, t: int) -> None:
        while env.t < t:
            env.step(self._zero_action())

    def _assert_load_invariants(self, loads: dict) -> None:
        self.assertEqual(set(loads), set(self.LOAD_NAMES))
        for name in self.LOAD_NAMES:
            with self.subTest(load=name):
                entry = loads[name]
                self.assertEqual(
                    set(entry), {"p_abc_kw", "q_abc_kvar", "p_total_kw", "q_total_kvar"}
                )
                self.assertEqual(entry["p_abc_kw"].shape, (3,))
                self.assertEqual(entry["q_abc_kvar"].shape, (3,))
                self.assertIsInstance(entry["p_total_kw"], float)
                self.assertIsInstance(entry["q_total_kvar"], float)
                self.assertAlmostEqual(entry["p_total_kw"], float(np.sum(entry["p_abc_kw"])))
                self.assertAlmostEqual(entry["q_total_kvar"], float(np.sum(entry["q_abc_kvar"])))
                self.assertTrue(np.all(entry["p_abc_kw"] >= 0.0))
                self.assertTrue(np.all(entry["q_abc_kvar"] >= 0.0))

        np.testing.assert_allclose(
            loads["office"]["p_abc_kw"] + loads["zoe"]["p_abc_kw"],
            loads["bus3"]["p_abc_kw"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            loads["office"]["q_abc_kvar"] + loads["zoe"]["q_abc_kvar"],
            loads["bus3"]["q_abc_kvar"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            loads["bus3"]["p_abc_kw"] + loads["wh1"]["p_abc_kw"] + loads["wh2"]["p_abc_kw"],
            loads["total"]["p_abc_kw"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            loads["bus3"]["q_abc_kvar"] + loads["wh1"]["q_abc_kvar"] + loads["wh2"]["q_abc_kvar"],
            loads["total"]["q_abc_kvar"],
            rtol=0.0,
            atol=1e-12,
        )

    def test_step_exposes_complete_effective_load_telemetry(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        _, reset_info = env.reset(seed=42)
        self.assertNotIn("loads", reset_info)

        _, _, _, _, info = env.step(self._zero_action())
        self._assert_load_invariants(info["loads"])

    def test_telemetry_matches_the_exact_snapshot_inputs(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.reset(seed=42)
        captured = {}
        original_solve = env._solve_electrical_snapshot

        def capture_snapshot_inputs(inputs):
            captured["inputs"] = inputs
            return original_solve(inputs)

        env._solve_electrical_snapshot = capture_snapshot_inputs
        _, _, _, _, info = env.step(self._zero_action())
        inputs = captured["inputs"]
        loads = info["loads"]
        for load_name, input_prefix in (("bus3", "bus3"), ("wh1", "wh1"), ("wh2", "wh2")):
            np.testing.assert_array_equal(
                loads[load_name]["p_abc_kw"], getattr(inputs, f"p_{input_prefix}_abc_kw")
            )
            np.testing.assert_array_equal(
                loads[load_name]["q_abc_kvar"], getattr(inputs, f"q_{input_prefix}_abc_kvar")
            )

    def test_info_arrays_are_independent_copies(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.reset(seed=42)
        source_hour_state = env._hour_state_cache
        self.assertIsNotNone(source_hour_state)
        office_before = source_hour_state["p_office_abc"].copy()
        bus3_before = source_hour_state["p_bus3_abc"].copy()

        _, _, _, _, info = env.step(self._zero_action())
        loads = info["loads"]
        self.assertFalse(np.shares_memory(loads["office"]["p_abc_kw"], source_hour_state["p_office_abc"]))
        self.assertFalse(np.shares_memory(loads["bus3"]["p_abc_kw"], source_hour_state["p_bus3_abc"]))

        loads["office"]["p_abc_kw"][0] += 1.0
        loads["bus3"]["p_abc_kw"][1] += 1.0
        np.testing.assert_array_equal(source_hour_state["p_office_abc"], office_before)
        np.testing.assert_array_equal(source_hour_state["p_bus3_abc"], bus3_before)




    def test_terminal_hour_contains_valid_load_telemetry(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        env.reset(seed=42)
        for _ in range(env.horizon):
            transition = env.step(self._zero_action())
            self._assert_load_invariants(transition[4]["loads"])

        _, _, terminated, truncated, info = transition
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["t"], env.horizon - 1)
        self._assert_load_invariants(info["loads"])

    def test_nominal_trajectory_is_unchanged_except_for_load_telemetry(self):
        first = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        second = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        obs_first, info_first = first.reset(seed=42)
        obs_second, info_second = second.reset(seed=42)
        np.testing.assert_array_equal(obs_first, obs_second)
        self._assert_value_equal(self, info_first, info_second)

        for _ in range(first.horizon):
            transition_first = first.step(self._zero_action())
            transition_second = second.step(self._zero_action())
            np.testing.assert_array_equal(transition_first[0], transition_second[0])
            self.assertEqual(transition_first[1:4], transition_second[1:4])
            first_info = {key: value for key, value in transition_first[4].items() if key not in {"loads", "episode_runtime"}}
            second_info = {key: value for key, value in transition_second[4].items() if key not in {"loads", "episode_runtime"}}
            self._assert_value_equal(self, first_info, second_info)


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
