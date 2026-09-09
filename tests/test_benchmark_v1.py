"""Acceptance checks for benchmark_v1. No policies trained or holdouts simulated.

Phase 2 approved the default c_unserved=1.0. Other coefficients in controlled
fixtures exercise parameterization without selecting scientific values.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from env.Ambiente_industrial import MicrorredIndustrial, compute_reward
from env.benchmark_metrics import EpisodeMetrics, truck_departure_event
from scripts.prepare_benchmark_v1_data import COLUMNS, extract_dated_weeks, profile_hash, write_deterministic_npz
from env.scenario_catalog import ScenarioCatalog, SOURCE_NAMES, SPLIT_NAMES


def make_env(**kwargs):
    return MicrorredIndustrial(c_unserved=2.0, extreme_events=None, **kwargs)


def assert_nested_equal(case, left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        case.assertEqual(left.keys(), right.keys())
        for key in left:
            # Only actual wall-clock runtime is nondeterministic.
            if key != 'episode_runtime':
                assert_nested_equal(case, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_equal(case, a, b)
    else:
        case.assertEqual(left, right)


class SplitAndCalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = ScenarioCatalog()
        cls.manifest = cls.catalog.manifest

    def test_disjoint_complete_partition_by_id_hash_and_date(self):
        for source in SOURCE_NAMES:
            records = self.manifest['sources'][source]['base_weeks']
            for attribute in ('base_id', 'profile_sha256', 'start'):
                if source == 'solar' and attribute == 'start':
                    continue
                sets = [{r[attribute] for r in records if r['split'] == split} for split in SPLIT_NAMES]
                for a, b in ((0, 1), (0, 2), (1, 2)):
                    self.assertFalse(sets[a] & sets[b], (source, attribute))
                self.assertEqual(set.union(*sets), {r[attribute] for r in records})
            for split in ('optuna_validation', 'final_test'):
                self.assertEqual(self.manifest['splits'][split]['sources'][source]['count'], 6)
        dates = [{r['start'] for src in COLUMNS for r in self.manifest['sources'][src]['base_weeks']
                  if r['split'] == split} for split in SPLIT_NAMES]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(dates[a] & dates[b])
        self.assertEqual([self.manifest['splits']['train']['sources'][s]['count'] for s in SOURCE_NAMES],
                         [69, 55, 85, 68])

    def test_included_dated_profiles_reconstruct_manifest_timestamps_and_hashes(self):
        # Derivative-level validation. The full original CSV is intentionally absent.
        for split in ('train', 'optuna_validation'):
            for source, column in COLUMNS.items():
                with np.load(self.catalog.pool_path(split, source), allow_pickle=False) as pool:
                    for base_id, week in zip(pool['base_ids'].tolist(), pool['weeks']):
                        saved = self.catalog._records[base_id]
                        start, end = pd.Timestamp(saved['start']), pd.Timestamp(saved['end'])
                        self.assertEqual((start.dayofweek, start.hour, end.dayofweek, end.hour), (0, 0, 6, 23))
                        self.assertEqual(end - start, pd.Timedelta(hours=167))
                        self.assertEqual(profile_hash(week), saved['profile_sha256'])
                        frame = pd.DataFrame({'timestamp': pd.date_range(start, periods=168, freq='h'), column: week})
                        records, values, discards = extract_dated_weeks(frame, source)
                        self.assertEqual(discards, [])
                        self.assertEqual(len(records), 1)
                        self.assertEqual(records[0]['base_id'], base_id)
                        self.assertEqual(records[0]['profile_sha256'], saved['profile_sha256'])
                        np.testing.assert_array_equal(values[0], week)

    def test_included_derived_hashes_solar_metadata_and_absent_holdout(self):
        for split in ('train', 'optuna_validation'):
            for source in SOURCE_NAMES:
                path = self.catalog.pool_path(split, source)
                self.assertEqual(path.parent.name, split)
            with np.load(self.catalog.pool_path(split, 'solar'), allow_pickle=False) as pool:
                for base_id, week in zip(pool['base_ids'].tolist(), pool['weeks']):
                    record = self.catalog._records[base_id]
                    self.assertIsNone(record['start'])
                    self.assertIsNone(record['end'])
                    self.assertEqual(base_id, f"solar-generic-{record['original_index']:04d}")
                    self.assertEqual(profile_hash(week), record['profile_sha256'])
        # Holdout identity and lineage remain metadata-only.
        for record in self.manifest['sources']['solar']['base_weeks']:
            self.assertIsNone(record['start'])
            self.assertIsNone(record['end'])
        self.assertFalse((CODE_DIR / 'data' / 'final_test').exists())

    def test_reference_scenarios_are_six_fixed_disjoint_compositions(self):
        for split in ('optuna_validation', 'final_test'):
            ids = self.catalog.reference_scenario_ids(split)
            self.assertEqual(len(set(ids)), 6)
            for index, scenario_id in enumerate(ids):
                choice = self.catalog.resolve(split, scenario_id)
                self.assertEqual(choice, self.catalog.resolve(split, f'{split}_{index+1:02d}'))
                starts = {self.catalog._records[choice.base_ids[s]]['start'] for s in COLUMNS}
                self.assertEqual(len(starts), 1)
                with self.assertRaises(ValueError):
                    self.catalog.resolve('train', scenario_id)

    def test_corrupt_manifest_or_pool_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scenario_splits.json'
            data = deepcopy(self.manifest)
            data['splits']['train']['sources']['office']['base_ids'][0] = data['splits']['final_test']['sources']['office']['base_ids'][0]
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                ScenarioCatalog(path)
            data = deepcopy(self.manifest)
            data['splits']['train']['sources']['office']['path'] = 'bad.npz'
            (path.parent / 'bad.npz').write_bytes(b'corrupt derived pool')
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                ScenarioCatalog(path).pool_path('train', 'office')

    def test_npz_generation_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('first.npz', 'second.npz')]
            for path in paths:
                write_deterministic_npz(path, weeks=np.arange(168.0)[None, :], base_ids=np.array(['example']))
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())


class ScenarioReproducibilityTests(unittest.TestCase):
    def test_gymnasium_api_contract_without_training(self):
        from gymnasium.utils.env_checker import check_env
        env = make_env()
        check_env(env, skip_render_check=True)

    def test_approved_scientific_default_and_explicit_override(self):
        for expected, kwargs in ((1.0, {}), (2.0, {'c_unserved': 2.0})):
            env = MicrorredIndustrial(split='train', **kwargs)
            _, info = env.reset(seed=1)
            self.assertEqual(env.c_unserved, expected)
            self.assertEqual(info['c_unserved_usd_per_kwh'], expected)
            info = env.step(np.zeros(12))[4]
            self.assertEqual(info['c_unserved_usd_per_kwh'], expected)
            self.assertEqual(env.get_episode_metrics()['c_unserved_usd_per_kwh'], expected)
        for invalid in (None, 0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                MicrorredIndustrial(c_unserved=invalid)
        with self.assertRaises(ValueError):
            make_env(split='validation')

    def test_constructor_seed_and_reset_scenario_roundtrip(self):
        first, second = make_env(seed=42), make_env(seed=42)
        a, ia = first.reset()
        b, ib = second.reset()
        assert_nested_equal(self, (a, ia), (b, ib))
        third = make_env(seed=999, scenario_id=ia['scenario_id'])
        c, ic = third.reset(seed=42)
        assert_nested_equal(self, (a, ia), (c, ic))
        d, idata = third.reset(seed=42, options={'scenario_id': ia['scenario_id']})
        assert_nested_equal(self, (a, ia), (d, idata))
        changed, im = third.reset(seed=43)
        self.assertEqual(ia['base_week_ids'], im['base_week_ids'])
        self.assertFalse(np.array_equal(a, changed))
        self.assertEqual(a.shape, (114,))
        self.assertEqual(first.action_space.shape, (12,))

    def test_only_train_pools_and_forecasts_are_accessible_to_train_reset(self):
        env = make_env()
        with self.assertRaises(AttributeError):
            env.split = 'final_test'
        allowed = env.scenario_catalog.manifest['splits']['train']['sources']
        self.assertTrue(all(split == 'train' for split, source in env.scenario_catalog._verified_paths))
        samplers = [('office', env.office_sampler), ('wh1', env.wh1_sampler), ('wh2', env.wh2_sampler),
                    ('solar', env.pv1.sampler), ('solar', env.pv2.sampler), ('solar', env.pv3.sampler)]
        for seed in range(20):
            _, info = env.reset(seed=seed)
            self.assertEqual(info['split'], 'train')
            for source, sampler in samplers:
                self.assertIn(info['base_week_ids'][source], allowed[source]['base_ids'])
                self.assertEqual(sampler.weeks_pool.shape[0], allowed[source]['count'])
                idx = allowed[source]['base_ids'].index(info['base_week_ids'][source])
                self.assertEqual(sampler.week_idx, idx)
                np.testing.assert_array_equal(sampler.P_real_ext[168:179], sampler.P_real_ext[:11])
                np.testing.assert_array_equal(sampler.P_forecast_ext, sampler.P_real_ext + sampler.forecast_noise_ext)
        before = env._get_obs().copy()
        test_id = env.scenario_catalog.reference_scenario_ids('final_test')[0]
        with self.assertRaises(ValueError):
            env.reset(seed=0, options={'scenario_id': test_id})
        with self.assertRaises(ValueError):
            env.reset(options={'split': 'final_test'})
        np.testing.assert_array_equal(before, env._get_obs())

    def test_full_168_hour_same_scenario_seed_actions_trajectory(self):
        first, second = make_env(), make_env()
        a, ia = first.reset(seed=2026)
        b, ib = second.reset(seed=2026, options={'scenario_id': ia['scenario_id']})
        assert_nested_equal(self, (a, ia), (b, ib))
        rng = np.random.default_rng(8801)
        for action in rng.uniform(-1, 1, (168, 12)).astype(np.float32):
            ta, tb = first.step(action), second.step(action)
            assert_nested_equal(self, ta, tb)
        self.assertTrue(ta[2])
        self.assertFalse(ta[3])
        self.assertEqual(first.get_episode_metrics(include_runtime=False), second.get_episode_metrics(include_runtime=False))


class TruckAndRewardTests(unittest.TestCase):
    def test_zero_at_or_above_target_and_proportional_deficit(self):
        events = [truck_departure_event(truck='truck5', departure_t=8, soc_departure=soc,
                                       target_soc=0.8, capacity_kwh=315, c_unserved=2)
                  for soc in (0.9, 0.8, 0.7, 0.6)]
        for event in events[:2]:
            self.assertEqual(event['energy_deficit_kwh'], 0)
            self.assertEqual(event['departure_penalty_usd'], 0)
            self.assertFalse(event['departure_violation'])
        self.assertAlmostEqual(events[2]['energy_deficit_kwh'], 31.5)
        self.assertAlmostEqual(events[2]['departure_penalty_usd'], 63.0)
        self.assertAlmostEqual(events[3]['departure_penalty_usd'], 126.0)
        self.assertGreater(events[3]['departure_penalty_usd'], events[2]['departure_penalty_usd'])

    def test_scheduled_departure_is_charged_once_without_forcing_soc(self):
        for soc, expected in ((0.9, 0.0), (0.8, 0.0), (0.7, 63.0), (0.6, 126.0)):
            with self.subTest(soc=soc):
                env = make_env()
                env.reset(seed=20)
                for _ in range(7):
                    self.assertFalse(env.step(np.zeros(12))[4]['truck_departures'])
                env.truck5.soc = soc  # controlled physical fixture immediately before departure
                _, reward, _, _, info = env.step(np.zeros(12))
                self.assertEqual(len(info['truck_departures']), 1)
                event = info['truck_departures'][0]
                self.assertEqual(event['truck'], 'truck5')
                self.assertEqual(event['departure_t'], 8)
                self.assertEqual(event['soc_departure'], soc)
                self.assertEqual(env.truck5.soc, soc)
                self.assertAlmostEqual(info['truck_departure_penalty_usd'], expected)
                self.assertAlmostEqual(event['energy_deficit_kwh'], info['truck_departure_soc_gaps']['truck5'] * env.truck5.C)
                self.assertAlmostEqual(reward, -info['energy_cost_usd'] - info['pf_penalty_usd'] - expected)
                self.assertEqual(env.get_episode_metrics()['truck_departure_count'], 1)
                after = env.step(np.zeros(12))[4]
                self.assertEqual(after['truck_departure_penalty_usd'], 0)
                self.assertEqual(env.get_episode_metrics()['truck_departure_count'], 1)

    def test_departure_uses_post_action_soc(self):
        env = make_env()
        env.reset(seed=20)
        for _ in range(7):
            env.step(np.zeros(12))
        env.truck5.soc = 0.6
        action = np.zeros(12)
        action[env.ACTION_TRUCK5_P] = -0.2  # external negative = consume/charge
        predicted = env.truck5.resolve_action_at(7, 0.2, 0.0)
        info = env.step(action)[4]
        event = info['truck_departures'][0]
        self.assertGreater(predicted.soc_next, 0.6)
        self.assertEqual(event['soc_departure'], predicted.soc_next)
        self.assertAlmostEqual(event['energy_deficit_kwh'], max(0, (0.8-predicted.soc_next)*315))

    def test_failed_departure_solve_does_not_charge_or_commit(self):
        env = make_env()
        env.reset(seed=20)
        for _ in range(7):
            env.step(np.zeros(12))
        before = (env.t, env.bess.E_kWh, env.truck5.soc, env.truck5.t, env.zoe_fleet.soc.copy())
        metrics = env.get_episode_metrics(include_runtime=False)
        with patch.object(env, '_solve_electrical_snapshot', return_value=SimpleNamespace(converged=False)):
            with self.assertRaises(RuntimeError):
                env.step(np.ones(12)*0.2)
        after = (env.t, env.bess.E_kWh, env.truck5.soc, env.truck5.t, env.zoe_fleet.soc.copy())
        assert_nested_equal(self, before, after)
        failed = env.get_episode_metrics(include_runtime=False)
        for key in ('steps', 'return_total', 'truck_departure_count', 'truck_departure_penalty_usd'):
            self.assertEqual(metrics[key], failed[key])
        self.assertEqual(failed['failed_solver_attempts'], 1)
        self.assertFalse(failed['all_solver_attempts_converged'])
        env.step(np.zeros(12))
        self.assertEqual(env.get_episode_metrics()['truck_departure_count'], 1)

    def test_pf_energy_formula_and_export_with_added_departure_term(self):
        for p in (-50, 50):
            for penalty in (0, 63, 126):
                result = compute_reward(p_grid_kw=p, price_raw=12, pf_phase=np.array([0.8, 0.95, 0.7]),
                                        q_pcc_kvar_phase=np.array([-10, 20, 30]), dt_h=0.5,
                                        lambda_pf=3, truck_departure_penalty_usd=penalty)
                self.assertAlmostEqual(result.energy_cost_usd, p*0.12*0.5)
                self.assertAlmostEqual(result.pf_penalty_usd, 3*0.5*(0.1*10+0.2*30))
                self.assertEqual(result.energy_cost_usd, result.operating_cost_usd)
                self.assertAlmostEqual(result.reward, -p*0.12*0.5 - 10.5 - penalty)


class EpisodeAccountingTests(unittest.TestCase):
    def test_controlled_voltage_pf_pv_and_energy_metrics(self):
        metrics = EpisodeMetrics(bess_soc=0.5, bess_energy_kwh=125, started_at=time.perf_counter())
        event = truck_departure_event(truck='truck5', departure_t=8, soc_departure=0.7,
                                      target_soc=0.8, capacity_kwh=315, c_unserved=2)
        info = dict(terminated=True, reward=-68, energy_cost_usd=-2, pf_penalty_usd=7,
                    truck_departure_penalty_usd=63, p_grid_kw=-4, p_losses_kw=2,
                    controls={name: dict(p_available_effective_kw=10, p_inj_kw=4) for name in ('pv1','pv2','pv3')},
                    voltages_pu={f'bus{i}': np.array([0.94,1.06,1]) for i in range(2,7)},
                    pf_phase=np.array([0.8,0.9,1]), truck_departures=[event],
                    max_lv_voltage_unbalance_pct=2.5, transformer_loading_pct=120, balance_residual_kw=-1e-7)
        info['controls']['bess'] = dict(soc_after=0.4)
        metrics.update(info, dt_h=0.5, bess_energy_kwh=100, elapsed_seconds=0.02)
        m = metrics.snapshot()
        self.assertEqual(m['export_kwh'], 2)
        self.assertEqual(m['import_kwh'], 0)
        self.assertEqual(m['losses_kwh'], 1)
        self.assertEqual((m['pv_available_kwh'], m['pv_utilized_kwh'], m['curtailment_kwh']), (15,6,9))
        self.assertEqual(m['voltage_violation_steps_phase'], [1,1,0])
        self.assertEqual(m['voltage_violation_node_phase_count'], 10)
        self.assertAlmostEqual(m['voltage_violation_max_pu'], 0.01)
        self.assertAlmostEqual(m['voltage_violation_sum_pu'], 0.1)
        self.assertAlmostEqual(m['voltage_violation_integral_pu_h'], 0.05)
        self.assertEqual(m['pf_violation_steps_phase'], [1,0,0])
        self.assertEqual(m['pf_mean_phase'], [0.8,0.9,1])
        self.assertEqual(m['transformer_loading_max_pct'], 120)
        self.assertEqual(m['vuf_max_pct'], 2.5)
        self.assertEqual(m['bess_energy_final_kwh'], 100)
        self.assertEqual(m['runtime']['step_seconds_total'], 0.02)

    def test_full_episode_aggregates_and_no_terminal_bess_term(self):
        env = make_env()
        _, reset_info = env.reset(seed=42)
        action = np.zeros(12)
        action[env.ACTION_BESS_P] = 1.0  # discharge; existing lower SOC clipping remains
        rows = []
        for _ in range(168):
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertAlmostEqual(reward, -info['energy_cost_usd']-info['pf_penalty_usd']-info['truck_departure_penalty_usd'])
            self.assertEqual(info['energy_cost_usd'], info['operating_cost_usd'])
            rows.append(info)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        np.testing.assert_array_equal(obs, np.zeros(114, dtype=np.float32))
        self.assertAlmostEqual(env.bess.soc, 0.08)
        self.assertEqual(reset_info['bess_energy_initial_kwh'], 125)
        m = env.get_episode_metrics(include_runtime=False)
        self.assertEqual(m, rows[-1]['episode_metrics'])
        self.assertEqual(m, env.get_episode_metrics(include_runtime=False))
        self.assertEqual(m['steps'], 168)
        self.assertTrue(m['complete'])
        self.assertEqual((m['bess_soc_initial'], m['bess_soc_final']), (0.5, 0.08))
        self.assertAlmostEqual(m['bess_energy_final_kwh'], 20)
        for key in ('energy_cost_usd','pf_penalty_usd','truck_departure_penalty_usd'):
            self.assertAlmostEqual(m[key], sum(r[key] for r in rows))
        self.assertAlmostEqual(m['return_total'], sum(r['reward'] for r in rows))
        self.assertAlmostEqual(m['import_kwh'], sum(max(r['p_grid_kw'],0) for r in rows))
        self.assertAlmostEqual(m['export_kwh'], sum(max(-r['p_grid_kw'],0) for r in rows))
        self.assertAlmostEqual(m['losses_kwh'], sum(r['p_losses_kw'] for r in rows))
        self.assertAlmostEqual(m['pv_available_kwh'], sum(r['controls'][p]['p_available_effective_kw'] for r in rows for p in ('pv1','pv2','pv3')))
        self.assertAlmostEqual(m['pv_utilized_kwh'], sum(r['controls'][p]['p_inj_kw'] for r in rows for p in ('pv1','pv2','pv3')))
        self.assertAlmostEqual(m['curtailment_kwh'], m['pv_available_kwh']-m['pv_utilized_kwh'])
        volts=np.array([[r['voltages_pu'][f'bus{i}'] for i in range(2,7)] for r in rows])
        np.testing.assert_array_equal(m['voltage_min_pu_phase'], volts.min(axis=(0,1)))
        np.testing.assert_array_equal(m['voltage_max_pu_phase'], volts.max(axis=(0,1)))
        self.assertEqual(m['voltage_violation_steps'], sum(bool(((v<0.95)|(v>1.05)).any()) for v in volts))
        pf=np.array([r['pf_phase'] for r in rows])
        np.testing.assert_allclose(m['pf_mean_phase'], pf.mean(axis=0))
        self.assertEqual(m['pf_violation_steps'], sum(bool((p<0.9).any()) for p in pf))
        events = [e for r in rows for e in r['truck_departures']]
        self.assertEqual(m['truck_departure_count'], 8)
        self.assertEqual([(e['truck'],e['departure_t']) for e in events],
                         [('truck5',8),('truck5',32),('truck6',32),('truck5',56),('truck5',80),('truck6',102),('truck5',104),('truck5',128)])
        self.assertEqual(m['truck_departures'], events)
        self.assertAlmostEqual(m['truck_energy_deficit_kwh'], sum(e['energy_deficit_kwh'] for e in events))
        self.assertEqual(m['vuf_max_pct'], max(r['max_lv_voltage_unbalance_pct'] for r in rows))
        self.assertEqual(m['transformer_loading_max_pct'], max(r['transformer_loading_pct'] for r in rows))
        self.assertEqual(m['converged_steps'], 168)
        self.assertEqual(m['failed_solver_attempts'], 0)
        self.assertGreater(rows[-1]['episode_runtime']['step_seconds_total'], 0)
        self.assertEqual(env.get_episode_metrics()['runtime'], rows[-1]['episode_runtime'])
        json.dumps(m, allow_nan=False)
        env.reset(seed=42)
        self.assertEqual(env.get_episode_metrics()['steps'], 0)
        self.assertEqual(env.get_episode_metrics()['truck_departure_count'], 0)


if __name__ == '__main__':
    unittest.main()
