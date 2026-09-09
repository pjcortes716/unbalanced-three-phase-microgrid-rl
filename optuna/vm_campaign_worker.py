"""One independent VM Python process per trial; never imported by the local smoke."""
from __future__ import annotations

# Permit both direct CLI execution and package imports from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
import weakref

from vm_campaign_config import (
    ARCHITECTURES, THREAD_VARIABLES, assert_campaign_science, assert_metrics_finite, can_prune, file_hash,
    make_pruner, read_config, single_thread_environment, storage, validation_pairs, verify_vm, write_atomic,
)


def run_job(job_path):
    single_thread_environment()  # before importing NumPy, Torch, OpenDSS, SB3 or Optuna
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    campaign = read_config(job["config_path"])
    os.chdir(job["trial_dir"])  # native OpenDSS/cache outputs stay private to this trial
    verify_vm(campaign, job["vm_hostname"])
    import numpy as np
    import optuna
    import opendssdirect as dss
    import psutil
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from threadpoolctl import threadpool_info
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from scripts.sac_experiment import (
        SACExperimentConfig, CheckedSAC, CsvSink, DataAccessAudit, TRAINING_FIELDS,
        aggregate_episode_rows, assert_model_finite, build_model, code_and_environment_manifest,
        compare_trajectories, evaluate_scenarios, make_environment,
    )
    from env.scenario_catalog import ScenarioCatalog
    out = Path(job["trial_dir"])
    process = psutil.Process()
    start = time.perf_counter()
    result = {"status": "RUNNING", "trial": job["number"], "stage": job["stage"],
              "pid": os.getpid(), "process_started_at": process.create_time(), "timesteps": 0,
              "wall_seconds": 0.0, "peak_rss_bytes": 0, "learning_seed": campaign["learning_seed"]}
    status_path = out / "status.json"
    write_atomic(status_path, result)
    env, model, audit, events = None, None, None, None
    sinks = []
    study = optuna.load_study(study_name=job["study_name"], storage=storage(campaign, job["campaign_dir"]),
                              pruner=make_pruner(campaign))
    trial = optuna.trial.Trial(study, job["trial_id"])  # version-pinned ask/tell handoff
    preflight = job["stage"] == "preflight"
    try:
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError("Torch thread limits were not applied.")
        pools = threadpool_info()
        if any(p["num_threads"] != 1 for p in pools if p.get("user_api") in ("blas", "openmp")):
            raise RuntimeError("BLAS/OpenMP thread limit mismatch.")
        result["thread_limits"] = {"torch_intra": torch.get_num_threads(), "torch_inter": torch.get_num_interop_threads(),
                                  "environment": {k: os.environ[k] for k in THREAD_VARIABLES}, "libraries": pools}
        dss.Basic.DataPath(str(out))
        dss.Basic.ClearAll()
        private_name = f"isolation_pid_{os.getpid()}"
        dss.Text.Command(f"New Circuit.{private_name}")
        if preflight:
            trial.set_user_attr("isolation_ready", {"pid": os.getpid(), "circuit": private_name})
            deadline = time.monotonic() + 120
            while sum("isolation_ready" in t.user_attrs for t in study.get_trials(deepcopy=False)) != 4:
                if time.monotonic() > deadline:
                    raise RuntimeError("Four-process storage/OpenDSS barrier timed out.")
                time.sleep(0.2)
            result["storage_concurrency_barrier"] = True
        time.sleep(1)
        if dss.Circuit.Name().lower() != private_name:
            raise RuntimeError("OpenDSS process isolation check failed.")
        result["opendss_private_circuit"] = dss.Circuit.Name()
        params = job["params"]
        budget = campaign["preflight_timesteps"] if preflight else campaign["max_timesteps"]
        config = SACExperimentConfig(
            seed=campaign["learning_seed"], net_arch=ARCHITECTURES[params["net_arch"]],
            learning_rate=params["learning_rate"], batch_size=params["batch_size"], gamma=params["gamma"],
            tau=params["tau"], learning_starts=params["learning_starts"], ent_coef=params["ent_coef"],
            total_timesteps=budget, allow_partial_last_episode=preflight,
            purpose="vm_concurrent_preflight" if preflight else "vm_optuna_trial")
        config.validate()
        assert_campaign_science(config)
        write_atomic(out / "params.json", params)
        write_atomic(out / "config.json", asdict(config))
        write_atomic(out / "environment_manifest.json", code_and_environment_manifest(config))
        if file_hash(Path(__file__).resolve().parents[1] / "data" / "scenario_splits.json") != campaign["scenario_manifest_sha256"]:
            raise RuntimeError("Scenario manifest changed from the approved campaign.")
        refs = ScenarioCatalog().reference_scenario_ids("optuna_validation")
        protocol = validation_pairs(refs, campaign)
        if preflight:
            protocol = protocol[:1]  # technical check only, not a tuning observation
        write_atomic(out / "validation_protocol.json", [{"scenario_id": s, "seed": seed} for s, seed in protocol])
        events = (out / "episode_events.jsonl").open("x", encoding="utf-8")
        class FiniteCsvSink(CsvSink):
            def write(self, row, **kwargs):
                assert_metrics_finite(row)
                return super().write(row, **kwargs)

        training = FiniteCsvSink(out / "training_log.csv", TRAINING_FIELDS)
        episodes = FiniteCsvSink(out / "episode_metrics.csv")
        validation = FiniteCsvSink(out / "validation_episodes.csv")
        history = FiniteCsvSink(out / "validation_history.csv")
        reload_sink = FiniteCsvSink(out / "validation_reloaded.csv")
        sinks.extend((training, episodes, validation, history, reload_sink))
        audit = DataAccessAudit(out / "data_access.jsonl")

        class Heartbeat(BaseCallback):
            def _on_step(self):
                now = time.perf_counter()
                if now - getattr(self, "last", 0) > 5:
                    result.update(timesteps=self.num_timesteps, wall_seconds=now-start,
                                  peak_rss_bytes=max(result["peak_rss_bytes"], process.memory_info().rss))
                    write_atomic(status_path, result)
                    self.last = now
                return True

        class CheckpointSink:
            def __init__(self, step):
                self.step = step
            def write(self, row):
                validation.write({"checkpoint": self.step, **row})

        with audit:
            with audit.phase("training"):
                env = make_environment(config, "train", sink=episodes, events_file=events, training=True)
                model = build_model(config, env)
            model._training_sink = training
            model._training_started_at = time.perf_counter()
            checkpoints = [budget] if preflight else campaign["checkpoints"]
            last_trace, last_value, pruned = None, None, False
            for step in checkpoints:
                with audit.phase("training"):
                    remaining = step - model.num_timesteps
                    model.learn(total_timesteps=remaining, reset_num_timesteps=False,
                                callback=Heartbeat(), log_interval=1)
                if model.num_timesteps != step:
                    raise RuntimeError("SAC did not reach the exact checkpoint.")
                assert_model_finite(model)
                rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
                try:
                    with audit.phase("validation_checkpoint"):
                        rows, last_trace = evaluate_scenarios(
                            model, config, protocol, sink=CheckpointSink(step), events_file=events, capture_first=True)
                finally:
                    random.setstate(rng[0])
                    np.random.set_state(rng[1])
                    torch.set_rng_state(rng[2])
                expected_count = 1 if preflight else 12
                if len(rows) != expected_count:
                    raise RuntimeError("Wrong number of validation trajectories.")
                metrics = aggregate_episode_rows(rows)
                last_value = float(np.mean([row["return_total"] for row in rows]))
                if not np.isfinite(last_value):
                    raise RuntimeError("NaN/Inf validation objective.")
                history.write({"checkpoint": step, "mean_validation_return": last_value,
                               "validation_return_std": float(np.std([r["return_total"] for r in rows])),
                               **{f"mean_{k}": v for k, v in metrics["means"].items()},
                               "voltage_violation_rate": metrics["voltage_violation_rate"],
                               "pf_violation_rate": metrics["pf_violation_rate"],
                               "pv_utilization_fraction": metrics["pv_utilization_fraction"],
                               "voltage_violation_max_pu": metrics["voltage_violation_max_pu"]})
                checkpoint = out / f"checkpoint_{step:06d}.zip"
                model.save(checkpoint)
                result.update(timesteps=step, latest_checkpoint=checkpoint.name,
                              mean_validation_return=last_value, wall_seconds=time.perf_counter()-start)
                write_atomic(status_path, result)
                trial.report(last_value, step)
                trial.set_user_attr("latest_checkpoint", step)
                if not preflight and can_prune(job["number"], step, campaign) and trial.should_prune():
                    pruned = True
                    break
            partial = None
            if model.num_timesteps % 168:
                partial = env.unwrapped.get_episode_metrics(include_runtime=False)
                partial.update(realization_seed=env.realization_seed,
                               stop_reason="explicit 2000-step concurrent technical preflight budget")
                write_atomic(out / "partial_episode_metrics.json", partial)
            model.save(out / "model_final.zip")
            result["gradient_updates"] = model._n_updates
            result["finite_update_checks"] = getattr(model, "_finite_update_checks", 0)
            if result["gradient_updates"] != model.num_timesteps - config.learning_starts:
                raise RuntimeError("Unexpected number of SAC gradient updates.")
            if result["finite_update_checks"] != result["gradient_updates"]:
                raise RuntimeError("Not every gradient update was checked for NaN/Inf.")
            result["replay_buffer_size"] = model.replay_buffer.size()
            reference = weakref.ref(model)
            del model
            model = None
            gc.collect()
            if reference() is not None:
                raise RuntimeError("Original model was not destroyed before save/load check.")
            loaded = CheckedSAC.load(out / "model_final.zip", device="cpu")
            assert_model_finite(loaded)
            with audit.phase("validation_reload"):
                _, loaded_trace = evaluate_scenarios(loaded, config, protocol[:1], sink=reload_sink,
                                                     events_file=events, capture_first=True)
            check = compare_trajectories(last_trace, loaded_trace)
            digest = hashlib.sha256()
            for key in sorted(last_trace):
                digest.update(key.encode())
                digest.update(np.ascontiguousarray(last_trace[key]).tobytes())
            del loaded
            gc.collect()
            # Check files after flushing them, including the final partial CSV buffer.
            expected_rows = ((training, result["gradient_updates"]), (episodes, env.completed_episodes),
                             (validation, len(checkpoints[:checkpoints.index(step)+1]) * expected_count),
                             (history, checkpoints.index(step)+1), (reload_sink, 1))
            for sink, expected in expected_rows:
                sink.file.flush()
                with sink.path.open(encoding="utf-8", newline="") as persisted:
                    if sum(1 for _ in csv.DictReader(persisted)) != expected:
                        raise RuntimeError(f"Persisted metric row count mismatch in {sink.path.name}.")
            result["metrics_verified"] = True
            access = audit.summary()
            if access["final_test_open_count"] or access["blocked_open_attempts"]:
                raise RuntimeError("Forbidden final_test dataset attempt.")
            result.update({
                "status": "PRUNED" if pruned else "COMPLETE", "objective": last_value,
                "wall_seconds": time.perf_counter()-start, "completed_episodes": env.completed_episodes,
                "partial_episode_steps": 0 if partial is None else partial["steps"],
                "peak_rss_bytes": max(result["peak_rss_bytes"], process.memory_info().rss),
                "cpu_seconds": sum(process.cpu_times()[:2]),
                "data_access": access, "final_test_accesses": 0,
                "opendss_failures": env.solver_failures, "nonfinite_incidents": 0,
                "save_load": check, "trajectory_sha256": digest.hexdigest(),
            })
            write_atomic(status_path, result)
            state = optuna.trial.TrialState.PRUNED if pruned else optuna.trial.TrialState.COMPLETE
            study.tell(job["number"], values=last_value if not pruned else None, state=state, skip_if_finished=True)
    except BaseException as error:
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc(),
                      wall_seconds=time.perf_counter()-start,
                      opendss_failures=max(0 if env is None else env.solver_failures,
                                           int(any(term in str(error).lower() for term in ("opendss", "converg", "dss")))),
                      nonfinite_incidents=int("NaN/Inf" in str(error)),
                      final_test_accesses=0 if audit is None else audit.summary()["final_test_open_count"],
                      data_access={} if audit is None else audit.summary())
        write_atomic(status_path, result)
        study.tell(job["number"], state=optuna.trial.TrialState.FAIL, skip_if_finished=True)
        raise
    finally:
        if env is not None:
            env.close()
        for sink in sinks:
            sink.close()
        if events is not None:
            events.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    run_job(parser.parse_args().job)

