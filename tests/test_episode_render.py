import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib
import numpy as np


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import env.Ambiente_industrial as Ambiente_industrial
from env.Ambiente_industrial import MicrorredIndustrial


class EpisodeRenderTests(unittest.TestCase):
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
                EpisodeRenderTests._assert_value_equal(test_case, actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            test_case.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                EpisodeRenderTests._assert_value_equal(test_case, actual_item, expected_item)
        else:
            test_case.assertEqual(actual, expected)

    def test_render_mode_contract_is_lazy_and_safe_by_default(self):
        self.assertEqual(MicrorredIndustrial.metadata, {"render_modes": ["episode"]})
        with self.assertRaisesRegex(ValueError, "render_mode.*episode"):
            MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, render_mode="human")

        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED)
        self.assertIsNone(env._render_history)
        self.assertIsNone(env._render_figure)
        env.reset(seed=41)
        env.step(self._zero_action())
        self.assertIsNone(env._render_history)
        with self.assertRaisesRegex(RuntimeError, "render_mode='episode'"):
            env.render()
        env.close()
        self.assertIsNone(env._render_figure)

        module_source = Path(Ambiente_industrial.__file__).read_text(encoding="utf-8")
        parsed = ast.parse(module_source)
        top_level_imports = [
            alias.name
            for node in parsed.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        ]
        self.assertFalse(any(name.startswith("matplotlib") for name in top_level_imports))

    def test_history_tracks_every_transition_and_resets(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, render_mode="episode")
        env.reset(seed=73)
        self.assertEqual(len(env._render_history), 0)
        with self.assertRaisesRegex(RuntimeError, "al menos un step"):
            env.render()

        for expected_t in range(env.horizon):
            _, reward, terminated, truncated, info = env.step(self._zero_action())
            self.assertEqual(len(env._render_history), expected_t + 1)
            record = env._render_history[-1]
            self.assertEqual(record["t"], expected_t)
            self.assertEqual(record["load_total_kw"], info["loads"]["total"]["p_total_kw"])
            self.assertEqual(record["pv1_kw"], info["controls"]["pv1"]["p_inj_kw"])
            self.assertEqual(record["pv2_kw"], info["controls"]["pv2"]["p_inj_kw"])
            self.assertEqual(record["pv3_kw"], info["controls"]["pv3"]["p_inj_kw"])
            self.assertEqual(record["bess_p_kw"], info["controls"]["bess"]["p_inj_kw"])
            self.assertEqual(record["bess_q_kvar"], info["controls"]["bess"]["q_inj_kvar"])
            self.assertEqual(record["bess_soc"], info["controls"]["bess"]["soc_after"])
            np.testing.assert_array_equal(record["pf_phase"], info["pf_phase"])
            self.assertEqual(record["price_raw"], info["price_raw"])
            self.assertEqual(record["reward"], reward)
            self.assertFalse(truncated)
            self.assertEqual(terminated, expected_t == env.horizon - 1)

        self.assertEqual(env._render_history[-1]["t"], 167)
        env.reset(seed=74)
        self.assertEqual(len(env._render_history), 0)




# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
