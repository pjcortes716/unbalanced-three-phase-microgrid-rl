"""Pruebas ejecutables de la interfaz OpenDSS corregida (ETAPA 1C-B)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import opendssdirect as dss


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import MicrorredIndustrial
from env.opendss_easy import DSSCircuit


class OpenDSSInterfaceTests(unittest.TestCase):
    R_OHM_PER_KM = 0.08
    X_OHM_PER_KM = 0.04

    def _new_circuit(self, name: str, with_feeder: bool = False) -> DSSCircuit:
        circuit = DSSCircuit(name=name, basekv=0.4, phases=3)
        circuit.connect_to_grid(bus="bus2", pu=1.0, angle=0.0, vsrc_name="Source")
        if with_feeder:
            circuit.add_unbalanced_line(
                "L23",
                "bus2",
                "bus3",
                length=0.05,
                units="km",
                r_per_phase=[self.R_OHM_PER_KM] * 3,
                x_per_phase=[self.X_OHM_PER_KM] * 3,
            )
        return circuit

    @staticmethod
    def _pcc_total(circuit: DSSCircuit):
        powers, _ = circuit.substation_phase_powers_fixed("bus2")
        return powers[powers["phase"] == "TOTAL"].iloc[0]

    def test_01_single_source(self):
        circuit = self._new_circuit("test_source")
        self.assertEqual([name.lower() for name in dss.Vsources.AllNames()], ["source"])
        dss.Circuit.SetActiveElement("Vsource.Source")
        self.assertEqual(dss.CktElement.BusNames()[0].split(".")[0].lower(), "bus2")
        self.assertEqual(dss.Vsources.Phases(), 3)
        self.assertAlmostEqual(dss.Vsources.BasekV(), 0.4, places=12)
        self.assertAlmostEqual(dss.Vsources.PU(), 1.0, places=12)
        self.assertAlmostEqual(dss.Vsources.AngleDeg(), 0.0, places=12)
        self.assertEqual(circuit.pcc_bus, "bus2")

    def test_02_linecode_diagonal_matrix(self):
        circuit = self._new_circuit("test_linecode")
        info = circuit.add_unbalanced_line(
            "L23",
            "bus2",
            "bus3",
            length=0.05,
            units="km",
            r_per_phase=[self.R_OHM_PER_KM] * 3,
            x_per_phase=[self.X_OHM_PER_KM] * 3,
        )
        self.assertTrue(circuit.solve())
        self.assertEqual(dss.Error.Number(), 0)

        dss.LineCodes.Name(info["linecode"])
        r_matrix = np.asarray(dss.LineCodes.Rmatrix(), dtype=float).reshape(3, 3)
        x_matrix = np.asarray(dss.LineCodes.Xmatrix(), dtype=float).reshape(3, 3)
        np.testing.assert_allclose(r_matrix, np.diag([0.08, 0.08, 0.08]), atol=1e-12)
        np.testing.assert_allclose(x_matrix, np.diag([0.04, 0.04, 0.04]), atol=1e-12)

        dss.Lines.Name("l23")
        self.assertAlmostEqual(dss.Lines.Length(), 0.05, places=12)
        self.assertAlmostEqual(r_matrix[0, 0] * dss.Lines.Length(), 0.004, places=12)
        self.assertAlmostEqual(x_matrix[0, 0] * dss.Lines.Length(), 0.002, places=12)

    def test_03_three_phase_load(self):
        circuit = self._new_circuit("test_load3", with_feeder=True)
        circuit.add_load("load3", "bus3", phases=3, kw=30.0, kvar=10.0)
        self.assertTrue(circuit.solve())
        dss.Loads.Name("load3")
        self.assertAlmostEqual(dss.Loads.kV(), 0.4, places=12)
        power = circuit.measure_power("Load.load3")
        self.assertAlmostEqual(power["P_kW"], 30.0, places=3)
        self.assertAlmostEqual(power["Q_kvar"], 10.0, places=3)

    def test_04_single_phase_load(self):
        circuit = self._new_circuit("test_load1", with_feeder=True)
        circuit.add_load("load1", "bus3.1", phases=1, kw=10.0, kvar=0.0)
        self.assertTrue(circuit.solve())
        dss.Loads.Name("load1")
        self.assertAlmostEqual(dss.Loads.kV(), 0.4 / np.sqrt(3.0), places=12)
        power = circuit.measure_power("Load.load1")
        self.assertAlmostEqual(power["P_kW"], 10.0, places=3)
        self.assertAlmostEqual(power["pairs"][0][0], 10.0, places=3)

    def test_05_unbalanced_phases(self):
        circuit = self._new_circuit("test_unbalanced", with_feeder=True)
        circuit.add_load("load_a", "bus3.1", phases=1, kw=10.0)
        circuit.add_load("load_b", "bus3.2", phases=1, kw=5.0)
        circuit.add_load("load_c", "bus3.3", phases=1, kw=0.0)
        self.assertTrue(circuit.solve())

        pcc, _ = circuit.substation_phase_powers_fixed("bus2")
        phase_p = pcc[pcc["phase"] != "TOTAL"]["P_kW"].to_numpy(dtype=float)
        self.assertGreater(phase_p[0], phase_p[1])
        self.assertGreater(phase_p[1], phase_p[2])

        dss.Circuit.SetActiveElement("Vsource.Source")
        source_currents = np.asarray(dss.CktElement.CurrentsMagAng(), dtype=float)[::2][:3]
        self.assertTrue(np.isfinite(source_currents).all())
        self.assertGreater(source_currents[0], source_currents[1])
        self.assertGreater(source_currents[1], source_currents[2])

        voltages = circuit.get_bus_voltages_df(["bus3"])
        self.assertTrue(np.isfinite(voltages["V_mag_V"]).all())
        self.assertTrue(np.isfinite(voltages["V_pu"]).all())

    def test_06_fixed_injection_signs(self):
        combinations = [(10.0, 4.0), (10.0, -4.0), (-10.0, 4.0), (-10.0, -4.0)]
        for index, (kw, kvar) in enumerate(combinations):
            with self.subTest(kw=kw, kvar=kvar):
                circuit = self._new_circuit(f"test_inj_{index}")
                circuit.add_fixed_injection("inj", "bus2.1", phases=1, kw=kw, kvar=kvar)
                self.assertTrue(circuit.solve())
                dss.Loads.Name("inj")
                self.assertAlmostEqual(dss.Loads.kV(), 0.4 / np.sqrt(3.0), places=12)
                self.assertAlmostEqual(dss.Loads.kW(), -kw, places=12)
                self.assertAlmostEqual(dss.Loads.kvar(), -kvar, places=12)
                power = circuit.measure_power("Load.inj")
                self.assertAlmostEqual(power["P_kW"], -kw, places=3)
                self.assertAlmostEqual(power["Q_kvar"], -kvar, places=3)

    def test_07_generator_voltage_and_injection(self):
        circuit = self._new_circuit("test_generator", with_feeder=True)
        circuit.add_generator("gen", "bus3", phases=3, kw=30.0, kvar=0.0)
        self.assertTrue(circuit.solve())
        dss.Generators.Name("gen")
        self.assertAlmostEqual(dss.Generators.kV(), 0.4, places=12)
        power = circuit.measure_power("Generator.gen")
        self.assertAlmostEqual(power["P_kW"], -30.0, places=3)

    def test_08_pcc_import_and_export(self):
        circuit = self._new_circuit("test_pcc_import", with_feeder=True)
        circuit.add_load("load", "bus3", phases=3, kw=30.0, kvar=0.0)
        self.assertTrue(circuit.solve())
        imported = self._pcc_total(circuit)
        self.assertGreater(imported["P_kW"], 30.0)

        circuit = self._new_circuit("test_pcc_export", with_feeder=True)
        circuit.add_load("load", "bus3", phases=3, kw=30.0, kvar=0.0)
        circuit.add_generator("gen", "bus3", phases=3, kw=50.0, kvar=0.0)
        self.assertTrue(circuit.solve())
        exported = self._pcc_total(circuit)
        self.assertLess(exported["P_kW"], 0.0)

    def test_09_physical_power_balance(self):
        circuit = self._new_circuit("test_balance", with_feeder=True)
        circuit.add_load("load", "bus3", phases=3, kw=30.0, kvar=10.0)
        circuit.add_generator("gen", "bus3", phases=3, kw=12.0, kvar=2.0)
        self.assertTrue(circuit.solve())
        balance = circuit.get_power_balance()
        self.assertLess(abs(balance["balance_residual_kW"]), 1e-6)
        self.assertLess(abs(balance["balance_residual_kvar"]), 1e-6)

    def test_10_voltage_per_unit(self):
        circuit = self._new_circuit("test_vpu", with_feeder=True)
        circuit.add_load("load", "bus3.1", phases=1, kw=10.0)
        self.assertTrue(circuit.solve())
        voltages = circuit.get_bus_voltages_df(["bus2", "bus3"])
        source_vpu = voltages[voltages["bus"] == "bus2"]["V_pu"].to_numpy(dtype=float)
        downstream_vpu = voltages[voltages["bus"] == "bus3"]["V_pu"].to_numpy(dtype=float)
        np.testing.assert_allclose(source_vpu, 1.0, atol=1e-3)
        self.assertTrue(np.isfinite(downstream_vpu).all())
        self.assertTrue(((downstream_vpu > 0.9) & (downstream_vpu < 1.1)).all())

    def test_11_empty_industrial_ring(self):
        environment = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, seed=7)
        circuit = environment.build_industrial_ring_400V()
        self.assertTrue(circuit.solve())
        self.assertEqual([name.lower() for name in dss.Vsources.AllNames()], ["source"])
        self.assertEqual(len(dss.Lines.AllNames()), 5)
        voltages = circuit.get_bus_voltages_df(["bus2", "bus3", "bus4", "bus5", "bus6"])
        self.assertTrue(np.isfinite(voltages["V_pu"]).all())
        np.testing.assert_allclose(voltages["V_pu"].to_numpy(dtype=float), 1.0, atol=1e-3)

    def test_12_unbalanced_industrial_ring(self):
        environment = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, seed=11)
        circuit = environment.build_industrial_ring_400V()
        circuit.add_load("load_b3_a", "bus3.1", phases=1, kw=10.0)
        circuit.add_load("load_b4_b", "bus4.2", phases=1, kw=5.0)
        circuit.add_load("load_b5_c", "bus5.3", phases=1, kw=2.0)
        self.assertTrue(circuit.solve())

        voltages = circuit.get_bus_voltages_df(["bus3", "bus4", "bus5", "bus6"])
        self.assertTrue(np.isfinite(voltages[["V_mag_V", "V_pu"]].to_numpy(dtype=float)).all())
        self.assertGreater(np.ptp(voltages["V_pu"].to_numpy(dtype=float)), 1e-7)

        pcc, _ = circuit.substation_phase_powers_fixed("bus2")
        phase_p = pcc[pcc["phase"] != "TOTAL"]["P_kW"].to_numpy(dtype=float)
        self.assertGreater(np.ptp(phase_p), 1.0)


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
