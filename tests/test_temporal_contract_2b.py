from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from env.Ambiente_industrial import MicrorredIndustrial
from env.PV_plant import WeeklyPVSampler
from env.oficinas import OfficeLoadSampler
from env.v2g_bus_5 import V2GTruck, availability_bus5_7x24, availability_bus6_7x24
from env.vehiculos_Zoe import ZoeFleetBus3


DATA_DIR = CODE_DIR / "data" / "train"


class TemporalContract2BTests(unittest.TestCase):
    def test_load_queries_are_non_mutating_and_future_only(self):
        sampler = OfficeLoadSampler(DATA_DIR / "office_weeks_pool.npz", seed=1)
        sampler.reset(seed=123)
        t_before = sampler.t

        first = sampler.value_at(17)
        second = sampler.value_at(17)
        forecast = sampler.forecast_at(17, horizon=11)
        third = sampler.value_at(17)

        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertEqual(sampler.t, t_before)
        self.assertEqual(forecast.shape, (11,))
        np.testing.assert_allclose(forecast, sampler.P_forecast_ext[18:29])
        self.assertEqual(forecast[0], sampler.P_forecast_ext[18])

        with self.assertRaises(IndexError):
            sampler.value_at(168)
        with self.assertRaises(IndexError):
            sampler.forecast_at(167, horizon=12)

    def test_pv_queries_are_non_mutating_and_ignore_legacy_clock(self):
        sampler = WeeklyPVSampler(
            DATA_DIR / "solar_weeks_pool.npz", seed=2, pmax_target_kw=74.25
        )
        sampler.reset(seed=456)
        t_before = sampler.t
        first = sampler.value_at(17)
        forecast = sampler.forecast_at(17)
        self.assertEqual(first, sampler.value_at(17))
        self.assertEqual(sampler.t, t_before)
        np.testing.assert_allclose(forecast, sampler.P_forecast_ext[18:29])

        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, seed=7)
        env.pv1.reset(seed=456)
        env.pv1.t = 40  # Reloj legacy deliberadamente distinto.
        plant_t_before = env.pv1.t
        sampler_t_before = env.pv1.sampler.t
        available = env.pv1.available_at(17)
        plant_forecast = env.pv1.forecast_at(17)
        self.assertEqual(available, env.pv1.sampler.value_at(17))
        np.testing.assert_allclose(plant_forecast, env.pv1.sampler.forecast_at(17))
        self.assertEqual(env.pv1.t, plant_t_before)
        self.assertEqual(env.pv1.sampler.t, sampler_t_before)

    def test_zoe_peek_does_not_consume_energy_and_advance_does_once(self):
        fleet = ZoeFleetBus3(seed=12, start_hour=0, end_hour=9)
        fleet.reset(seed=999)
        fleet.prepare_interval(0)
        soc_before = fleet.soc.copy()
        eac_before = fleet.Eac_rem.copy()
        t_before = fleet.t

        request_1 = fleet.get_power_request(0)
        request_2 = fleet.get_power_request(0)
        np.testing.assert_allclose(request_1, request_2)
        np.testing.assert_allclose(fleet.soc, soc_before)
        np.testing.assert_allclose(fleet.Eac_rem, eac_before)
        self.assertEqual(fleet.t, t_before)

        done = fleet.advance_interval(0, request_1)
        self.assertFalse(done)
        self.assertEqual(fleet.t, 1)
        self.assertTrue(np.all(fleet.Eac_rem <= eac_before))
        self.assertTrue(np.all(fleet.soc >= soc_before))

    def test_zoe_last_charge_hour_is_limited_without_query_mutation(self):
        fleet = ZoeFleetBus3(start_hour=0, end_hour=9)
        fleet.reset()
        fleet.t = 1
        fleet.connected[:] = True
        fleet.Eac_rem[:] = 10.0
        fleet.soc[:] = fleet.soc_target - fleet.eta * 10.0 / fleet.C
        fleet.prepare_interval(1)

        request = fleet.get_power_request(1)
        np.testing.assert_allclose(request, [10.0, 10.0])
        np.testing.assert_allclose(fleet.Eac_rem, [10.0, 10.0])
        fleet.advance_interval(1, request)
        np.testing.assert_allclose(fleet.Eac_rem, [0.0, 0.0])
        np.testing.assert_allclose(fleet.soc, [fleet.soc_target, fleet.soc_target])

    def _advance_truck_to(self, truck: V2GTruck, target_t: int) -> None:
        while truck.t < target_t:
            truck.apply_action_at(truck.t, 0.0, 0.0)
            truck.advance_interval(truck.t)

    def test_truck_bus5_frontiers_return_schedule_for_current_hour(self):
        truck = V2GTruck(availability_bus5_7x24, seed=4)
        truck.reset(seed=4)
        self._advance_truck_to(truck, 7)
        self.assertEqual(truck.state_at(7)[0], 1.0)
        truck.apply_action_at(7, 0.0, 0.0)
        _, done = truck.advance_interval(7)
        self.assertFalse(done)
        self.assertEqual(truck.t, 8)
        self.assertEqual(truck.state_at(8)[0], 0.0)

        self._advance_truck_to(truck, 16)
        self.assertEqual(truck.state_at(16)[0], 0.0)
        truck.apply_action_at(16, 0.0, 0.0)
        truck.advance_interval(16)
        self.assertEqual(truck.t, 17)
        self.assertEqual(truck.state_at(17)[0], 1.0)
        self.assertEqual(truck.get_state().shape, (5,))
        self.assertTrue(np.all((0.0 <= truck.get_state()) & (truck.get_state() <= 1.0)))

    def test_truck_bus6_frontiers_return_schedule_for_current_hour(self):
        truck = V2GTruck(availability_bus6_7x24, seed=5)
        truck.reset(seed=5)
        self._advance_truck_to(truck, 16)
        self.assertEqual(truck.state_at(16)[0], 0.0)
        truck.apply_action_at(16, 0.0, 0.0)
        truck.advance_interval(16)
        self.assertEqual(truck.state_at(17)[0], 1.0)

        self._advance_truck_to(truck, 31)
        self.assertEqual(truck.state_at(31)[0], 1.0)
        truck.apply_action_at(31, 0.0, 0.0)
        truck.advance_interval(31)
        self.assertEqual(truck.state_at(32)[0], 0.0)

    def test_same_seed_reproduces_load_pv_zoe_and_truck(self):
        load_a = OfficeLoadSampler(DATA_DIR / "office_weeks_pool.npz", seed=0)
        load_b = OfficeLoadSampler(DATA_DIR / "office_weeks_pool.npz", seed=0)
        load_a.reset(seed=123)
        load_b.reset(seed=123)
        self.assertEqual(load_a.original_week_idx, load_b.original_week_idx)
        np.testing.assert_allclose(load_a.P_real_ext, load_b.P_real_ext)
        np.testing.assert_allclose(load_a.P_forecast_ext, load_b.P_forecast_ext)

        pv_a = WeeklyPVSampler(DATA_DIR / "solar_weeks_pool.npz", seed=0, pmax_target_kw=10.0)
        pv_b = WeeklyPVSampler(DATA_DIR / "solar_weeks_pool.npz", seed=0, pmax_target_kw=10.0)
        pv_a.reset(seed=456)
        pv_b.reset(seed=456)
        self.assertEqual(pv_a.week_idx, pv_b.week_idx)
        np.testing.assert_allclose(pv_a.P_real_ext, pv_b.P_real_ext)
        np.testing.assert_allclose(pv_a.P_forecast_ext, pv_b.P_forecast_ext)

        zoe_a = ZoeFleetBus3(start_hour=0, seed=0)
        zoe_b = ZoeFleetBus3(start_hour=0, seed=0)
        zoe_a.reset(seed=789)
        zoe_b.reset(seed=789)
        zoe_a.prepare_interval(0)
        zoe_b.prepare_interval(0)
        np.testing.assert_allclose(zoe_a.soc, zoe_b.soc)
        np.testing.assert_allclose(zoe_a.get_power_request(0), zoe_b.get_power_request(0))

        truck_a = V2GTruck(availability_bus5_7x24, seed=0)
        truck_b = V2GTruck(availability_bus5_7x24, seed=0)
        np.testing.assert_allclose(truck_a.reset(seed=321), truck_b.reset(seed=321))

    def test_environment_rng_reproduces_pf_and_phase_weights(self):
        env_a = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, seed=2468)
        env_b = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, seed=2468)
        pf_a = [
            env_a.sample_power_factor(0.80, 0.995),
            env_a.sample_power_factor(0.57, 0.88),
            env_a.sample_power_factor(0.57, 0.88),
        ]
        pf_b = [
            env_b.sample_power_factor(0.80, 0.995),
            env_b.sample_power_factor(0.57, 0.88),
            env_b.sample_power_factor(0.57, 0.88),
        ]
        np.testing.assert_allclose(pf_a, pf_b)
        np.testing.assert_allclose(
            env_a.sample_phase_weights(0.20, 0.40),
            env_b.sample_phase_weights(0.20, 0.40),
        )

    def test_environment_contract_layout_is_exactly_114(self):
        env = MicrorredIndustrial(c_unserved=TEST_C_UNSERVED, seed=11)
        self.assertEqual(env.n_obs, 114)
        self.assertEqual(env.observation_space.shape, (114,))
        self.assertEqual(env.OBS_TIME, slice(0, 2))
        self.assertEqual(env.OBS_BUS3_LOAD, slice(2, 8))
        self.assertEqual(env.OBS_TRUCK5, slice(90, 95))
        self.assertEqual(env.OBS_TRUCK6, slice(95, 100))
        self.assertEqual(env.OBS_ZOE_SOC, slice(100, 102))
        self.assertEqual(env.OBS_CURRENT_PRICE, slice(102, 103))
        self.assertEqual(env.OBS_FORECAST_PRICE, slice(103, 114))
        self.assertEqual(env.ACTION_NAMES[0], "pv1_utilization")
        self.assertEqual(env.ACTION_NAMES[-1], "truck6_q")


# Fixture numérica; NO es una calibración científica.
TEST_C_UNSERVED = 1.0

if __name__ == "__main__":
    unittest.main()
