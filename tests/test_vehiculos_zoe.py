from pathlib import Path
import sys
import unittest


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.vehiculos_Zoe import ZoeFleetBus3


class ZoeFleetBus3Tests(unittest.TestCase):
    def test_last_charge_hour_uses_remaining_ac_energy(self):
        fleet = ZoeFleetBus3(start_hour=0, end_hour=9)
        fleet.reset()

        # Hora posterior a la llegada, dentro de la ventana de carga.
        fleet.t = 1
        fleet.connected[:] = True
        fleet.Eac_rem[:] = 10.0
        fleet.soc[:] = fleet.soc_target - fleet.eta * 10.0 / fleet.C

        p_ev0, p_ev1, _ = fleet.step()

        self.assertEqual(p_ev0, 10.0)
        self.assertEqual(p_ev1, 10.0)
        self.assertTrue((fleet.Eac_rem == 0.0).all())
        self.assertTrue((fleet.soc <= fleet.soc_target).all())
        self.assertTrue((fleet.soc == fleet.soc_target).all())


if __name__ == "__main__":
    unittest.main()
