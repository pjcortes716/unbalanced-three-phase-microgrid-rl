"""Regresiones del transformador de producción 15/0.4 kV (caso T2)."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import opendssdirect as dss


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import ElectricalSnapshotInputs, MicrorredIndustrial


def _q_from_pf(p_kw: float, pf: float) -> float:
    return float(p_kw * math.tan(math.acos(pf)))


def _building_inputs(
    weights: np.ndarray,
    *,
    bess: tuple[float, float] = (0.0, 0.0),
    truck5: tuple[float, float] = (0.0, 0.0),
    truck6: tuple[float, float] = (0.0, 0.0),
) -> ElectricalSnapshotInputs:
    """Demanda extrema del diagnóstico; Zoe A/B absorben 22 kW en bus3."""
    office_p, wh1_p, wh2_p = 18.74, 65.66, 76.12
    return ElectricalSnapshotInputs(
        p_bus3_abc_kw=office_p * weights + np.array([22.0, 22.0, 0.0]),
        q_bus3_abc_kvar=_q_from_pf(office_p, 0.80) * weights,
        p_wh1_abc_kw=wh1_p * weights,
        q_wh1_abc_kvar=_q_from_pf(wh1_p, 0.57) * weights,
        p_wh2_abc_kw=wh2_p * weights,
        q_wh2_abc_kvar=_q_from_pf(wh2_p, 0.57) * weights,
        bess_p_inj_kw=bess[0],
        bess_q_inj_kvar=bess[1],
        truck5_p_inj_kw=truck5[0],
        truck5_q_inj_kvar=truck5[1],
        truck6_p_inj_kw=truck6[0],
        truck6_q_inj_kvar=truck6[1],
    )


class TransformerProductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, seed=42)

    @staticmethod
    def _vmin(result) -> float:
        return min(float(value) for values in result.voltages_vpu.values() for value in values)

    @staticmethod
    def _vmax(result) -> float:
        return max(float(value) for values in result.voltages_vpu.values() for value in values)

    def test_01_explicit_t2_connection_and_multivoltage_bases(self):
        circuit = self.env.build_industrial_network_with_transformer()
        self.assertTrue(circuit.solve())
        voltage_df = circuit.get_bus_voltages_df(["bus1", "bus2"])
        for bus in ("bus1", "bus2"):
            values = voltage_df[voltage_df["bus"] == bus]["V_pu"].to_numpy(dtype=float)
            self.assertEqual(values.shape, (3,))
            np.testing.assert_allclose(values, 1.0, atol=1e-7)

        dss.Circuit.SetActiveElement("Transformer.tx")
        self.assertEqual(dss.CktElement.NodeOrder(), [1, 2, 3, 0, 1, 2, 3, 0])
        self.assertEqual(dss.Circuit.NumNodes(), 18)  # no distributed neutral node
        dss.Transformers.Name("tx")
        self.assertAlmostEqual(dss.Transformers.Xhl(), math.sqrt(35.0), places=9)
        for winding, expected_kv in ((1, 15.0), (2, 0.4)):
            dss.Transformers.Wdg(winding)
            self.assertFalse(dss.Transformers.IsDelta())
            self.assertAlmostEqual(dss.Transformers.kV(), expected_kv, places=9)
            self.assertAlmostEqual(dss.Transformers.R(), 0.5, places=9)
            self.assertAlmostEqual(dss.Transformers.Rneut(), -1.0, places=9)
            self.assertAlmostEqual(dss.Transformers.Xneut(), 0.0, places=9)

    def test_02_manual_150_plus_j100_reports_pcc_transformer_and_lv_losses(self):
        result = self.env._solve_electrical_snapshot(
            ElectricalSnapshotInputs(
                p_wh1_abc_kw=np.full(3, 50.0),
                q_wh1_abc_kvar=np.full(3, 100.0 / 3.0),
            )
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.diagnostics["pcc_bus"], "bus1")
        np.testing.assert_allclose(result.bus2_vpu, 0.99043252, atol=2e-6)
        np.testing.assert_allclose(result.voltages_vpu["bus5"], 0.97973878, atol=2e-6)
        self.assertAlmostEqual(result.pcc["P_total_kW"], 152.19674699, places=5)
        self.assertAlmostEqual(result.pcc["Q_total_kvar"], 103.39131460, places=5)
        self.assertAlmostEqual(result.transformer_loading_percent, 22.99918943, places=5)
        self.assertGreater(result.transformer_p_losses_kw, 0.0)
        self.assertGreater(result.transformer_q_losses_kvar, 0.0)
        self.assertGreater(result.p_losses_kw, result.transformer_p_losses_kw)
        self.assertLess(abs(result.power_balance["balance_residual_kW"]), 1e-5)

    def test_03_t2_a2_max_active_matches_constant_power_reference(self):
        weights = np.array([0.40, 0.30, 0.30])
        result = self.env._solve_electrical_snapshot(
            _building_inputs(weights, bess=(-180.0, 0.0), truck5=(-150.0, 0.0), truck6=(-150.0, 0.0))
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(self._vmin(result), 0.9277856831, delta=5e-4)
        self.assertAlmostEqual(result.voltages_vpu["bus2"][0], 0.9673071799, delta=5e-4)
        self.assertAlmostEqual(result.transformer_loading_percent, 95.65062516, delta=0.05)
        self.assertAlmostEqual(result.pcc["P_total_kW"], 714.6811913, delta=0.05)

    def test_04_t2_worst_pq_matches_constant_power_reference(self):
        weights = np.array([0.40, 0.40, 0.20])
        result = self.env._solve_electrical_snapshot(
            _building_inputs(
                weights,
                bess=(-159.9511018778908, -82.55691980717668),
                truck5=(-133.2324970655224, -68.91372668551284),
                truck6=(-132.9911315699737, -69.3783750439425),
            )
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(self._vmin(result), 0.9067990087, delta=5e-4)
        self.assertAlmostEqual(result.transformer_loading_percent, 104.73651037, delta=0.05)
        self.assertAlmostEqual(result.pcc["Q_total_kvar"], 506.2344498, delta=0.05)

    def test_05_t2_export_matches_constant_power_reference(self):
        env = self.env
        p_office, p_wh1, p_wh2 = 2.9189355021990737, 0.0, 7.064092544987147
        pv = (
            (74.25, 66.98460644058454),
            (75.58, 65.48025351203216),
            (7.75, 6.319612329882269),
        )
        weights = np.full(3, 1.0 / 3.0)
        result = env._solve_electrical_snapshot(
            ElectricalSnapshotInputs(
                p_bus3_abc_kw=p_office * weights,
                q_bus3_abc_kvar=_q_from_pf(p_office, env.PF_OFFICE_RANGE[1]) * weights,
                p_wh1_abc_kw=p_wh1 * weights,
                q_wh1_abc_kvar=_q_from_pf(p_wh1, env.PF_WAREHOUSE_RANGE[1]) * weights,
                p_wh2_abc_kw=p_wh2 * weights,
                q_wh2_abc_kvar=_q_from_pf(p_wh2, env.PF_WAREHOUSE_RANGE[1]) * weights,
                pv1_p_inj_kw=pv[0][0], pv1_q_inj_kvar=pv[0][1],
                pv2_p_inj_kw=pv[1][0], pv2_q_inj_kvar=pv[1][1],
                pv3_p_inj_kw=pv[2][0], pv3_q_inj_kvar=pv[2][1],
                bess_p_inj_kw=160.59456247503326, bess_q_inj_kvar=81.2981334561418,
                truck5_p_inj_kw=133.88787276781883, truck5_q_inj_kvar=67.63163110341462,
                truck6_p_inj_kw=133.76962944078213, truck6_q_inj_kvar=67.8652063967674,
            )
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(self._vmax(result), 1.0634982768, delta=5e-4)
        self.assertAlmostEqual(result.transformer_loading_percent, 79.44782479, delta=0.05)
        self.assertLess(result.pcc["P_total_kW"], 0.0)
        self.assertAlmostEqual(result.pcc["P_total_kW"], -553.2591584, delta=0.05)


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
