"""Persistent, process-only Optuna coordinator. Running on the local PC is blocked."""
from __future__ import annotations

# Permit both direct CLI execution and package imports from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

from vm_campaign_config import (
    CODE_DIR, file_hash, make_pruner, preflight_parameters, read_config, sampler_for_number,
    single_thread_environment, storage, suggest_parameters, verify_vm, write_atomic,
)

TERMINAL = {"COMPLETE", "PRUNED", "FAIL"}
RESOURCE_FIELDS = ("epoch", "stage", "active_workers", "machine_cpu_percent",
                   "machine_used_ram_bytes", "workers_rss_bytes")


def read_status(directory):
    path = Path(directory) / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def process_alive(record):
    import psutil
    try:
        proc = psutil.Process(record["pid"])
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE and abs(
            proc.create_time() - record["process_started_at"]) < 0.01
    except (psutil.Error, KeyError, TypeError):
        return False


def scheduling_window(trials, config, preflight=False):
    """Barrier: all 12 warm-up trials must be terminal before any TPE trial."""
    if preflight:
        return config["preflight_workers"], config["preflight_workers"], "preflight"
    if len(trials) < config["warmup_trials"] or any(
            t.state.name not in TERMINAL for t in trials[:config["warmup_trials"]]):
        return config["warmup_trials"], config["warmup_workers"], "warmup"
    return config["total_trials"], config["tpe_workers"], "tpe"


class CoordinatorLock:
    def __init__(self, root, config):
        self.path, self.config = root / "coordinator.lock", config
        self.connection = self.engine = None
        self.owned = False

    def __enter__(self):
        import psutil
        if self.path.exists():
            previous = json.loads(self.path.read_text(encoding="utf-8"))
            if previous.get("hostname") != socket.gethostname() or process_alive(previous):
                raise RuntimeError("Another coordinator owns this campaign.")
            self.path.unlink()  # one stale lock file, never a recursive removal
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "process_started_at": psutil.Process().create_time(),
                       "hostname": socket.gethostname()}, stream)
        self.owned = True
        try:
            if self.config["storage"] == "postgresql":
                from sqlalchemy import create_engine, text
                self.engine = create_engine(os.environ["OPTUNA_STORAGE_URL"], pool_size=1, max_overflow=0)
                self.connection = self.engine.connect()
                self.key = int.from_bytes(hashlib.sha256(
                    self.config["study_name"].encode()).digest()[:8], "big") % (2**63)
                if not self.connection.execute(text("SELECT pg_try_advisory_lock(:key)"),
                                               {"key": self.key}).scalar():
                    raise RuntimeError("Another coordinator owns this PostgreSQL study.")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.connection is not None:
            self.connection.close()  # advisory lock lasts only for this connection
        if self.engine is not None:
            self.engine.dispose()
        if self.owned:
            self.path.unlink(missing_ok=True)


def summarize(study, root, metadata, resource_rows):
    trials = study.get_trials(deepcopy=False)
    statuses = [read_status(root / "trials" / f"trial_{t.number:04d}") for t in trials]
    complete = sorted((t for t in trials if t.state.name == "COMPLETE"),
                      key=lambda t: t.value, reverse=True)
    top = [{"trial": t.number, "mean_validation_return": t.value, "params": t.params,
            "checkpoint_returns": {str(k): v for k, v in sorted(t.intermediate_values.items())},
            "wall_seconds": read_status(root / "trials" / f"trial_{t.number:04d}").get("wall_seconds"),
            "model_path": str(root / "trials" / f"trial_{t.number:04d}" / "model_final.zip")}
           for t in complete[:10]]
    finalists, signatures = [], set()
    for t in complete:
        signature = json.dumps(t.params, sort_keys=True)
        if signature not in signatures:
            signatures.add(signature)
            finalists.append({"trial": t.number, "params": t.params, "mean_validation_return": t.value})
        if len(finalists) == 5:
            break
    for entry in top:
        values = list(entry["checkpoint_returns"].values())
        tail = values[-3:]
        entry["checkpoint_evolution"] = {
            "last_minus_first": values[-1] - values[0] if values else None,
            "range_last_three_checkpoints": max(tail) - min(tail) if tail else None,
            "interpretation": "Descriptive learning trajectory on fixed validation realizations; not multiseed robustness.",
        }
    preflight = [read_status(p) for p in sorted((root / "preflight").glob("trial_*"))]
    all_statuses = statuses + preflight
    result = {
        "finished": True, "study_name": study.study_name,
        "trials_allocated": len(trials),
        "counts": {state: sum(t.state.name == state for t in trials)
                   for state in ("COMPLETE", "PRUNED", "FAIL", "RUNNING")},
        "wall_seconds_total": time.time() - metadata["created_at_epoch"],
        "wall_time_scope": "Since first coordinator start, including preflight and any downtime/resume gaps.",
        "peak_machine_cpu_percent": max((float(r["machine_cpu_percent"]) for r in resource_rows), default=0),
        "peak_machine_used_ram_bytes": max((int(r["machine_used_ram_bytes"]) for r in resource_rows), default=0),
        "peak_workers_rss_bytes": max((int(r["workers_rss_bytes"]) for r in resource_rows), default=0),
        "peak_active_workers": max((int(r["active_workers"]) for r in resource_rows), default=0),
        "resource_sampling_seconds": 5,
        "top_10": top, "best_parameters": top[0]["params"] if top else None,
        "finalists": finalists,
        "finalist_selection": "Up to five distinct completed configurations ranked only by the final 12-episode mean return.",
        "opendss_failures": sum(s.get("opendss_failures", 0) for s in all_statuses),
        "nonfinite_incidents": sum(s.get("nonfinite_incidents", 0) for s in all_statuses),
        "final_test_accesses": sum(s.get("final_test_accesses", 0) for s in all_statuses),
        "blocked_data_access_attempts": sum(len(s.get("data_access", {}).get("blocked_open_attempts", [])) for s in all_statuses),
        "incident_scope": "Optimization and concurrent preflight; failures with missing telemetry are listed separately.",
        "incomplete_failure_telemetry_trials": [s.get("trial") for s in statuses
                                              if s.get("status") == "FAIL" and s.get("telemetry_incomplete")],
    }
    if result["final_test_accesses"] != 0:
        raise RuntimeError("Campaign violated the final_test access invariant.")
    write_atomic(root / "global_summary.json", result)
    with (root / "top_10.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("trial", "mean_validation_return", "wall_seconds", "params", "checkpoint_returns", "checkpoint_evolution", "model_path"))
        writer.writeheader()
        for entry in top:
            writer.writerow({**entry, "params": json.dumps(entry["params"]),
                             "checkpoint_returns": json.dumps(entry["checkpoint_returns"]),
                             "checkpoint_evolution": json.dumps(entry["checkpoint_evolution"])})
    fence = chr(96) * 3
    lines = ["# Resultados Optuna en la VM", "",
             f"Trials: {result['counts']}. Wall time total: {result['wall_seconds_total']:.1f} s.",
             f"CPU máxima muestreada: {result['peak_machine_cpu_percent']:.1f}% de la máquina. "
             f"RAM usada máxima: {result['peak_machine_used_ram_bytes']/1e9:.2f} GB; "
             f"RSS agregado máximo de workers: {result['peak_workers_rss_bytes']/1e9:.2f} GB.",
             f"Fallos OpenDSS: {result['opendss_failures']}; NaN/Inf: {result['nonfinite_incidents']}; "
             f"final_test accesses: {result['final_test_accesses']}.", "",
             "## Top 10 y evolución por checkpoints", "",
             "| Trial | Return final | 16800 | 33600 | 50400 | 67200 | 84000 | 100800 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in top:
        values = [str(entry["trial"]), f"{entry['mean_validation_return']:.6f}"] + [
            f"{entry['checkpoint_returns'][str(k)]:.6f}" if str(k) in entry["checkpoint_returns"] else "—"
            for k in (16800, 33600, 50400, 67200, 84000, 100800)]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "La tabla describe evolución con las mismas realizaciones; no demuestra estabilidad multisemilla.",
              "", "## Parámetros de la mejor configuración", "", fence + "json",
              json.dumps(result["best_parameters"], indent=2), fence, "",
              "## Configuraciones propuestas para confirmación futura", "", fence + "json",
              json.dumps(finalists, indent=2), fence, "",
              "Selección por return final, sin filtros adicionales. No se ha ejecutado confirmación."]
    if len(finalists) < 3:
        lines.append("No hay tres configuraciones completas distintas; no se inventan finalistas.")
    if result["incomplete_failure_telemetry_trials"]:
        lines.append("Hay procesos interrumpidos con telemetría incompleta; revisar global_summary.json.")
    (root / "OPTUNA_VM_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def supervise(study, root, config, config_path, hostname, resource_writer, resource_rows, *, preflight=False):
    import optuna
    import psutil
    directory = root / ("preflight" if preflight else "trials")
    directory.mkdir(exist_ok=True)
    cap = config["preflight_workers"] if preflight else config["total_trials"]
    children = {}
    peak = 0
    try:
        while True:
            for number, (child, log) in list(children.items()):
                if child.poll() is not None:
                    log.close()
                    del children[number]
            trials = study.get_trials(deepcopy=False)
            if len(trials) > cap:
                raise RuntimeError("The persistent study exceeds the approved trial cap.")
            active = []
            for frozen in trials:
                trial_dir = directory / f"trial_{frozen.number:04d}"
                status = read_status(trial_dir)
                pid_record = status
                if "pid" not in status and (trial_dir / "process.json").exists():
                    pid_record = json.loads((trial_dir / "process.json").read_text(encoding="utf-8"))
                alive = process_alive(pid_record)
                if alive:
                    active.append(pid_record["pid"])
                if frozen.state.name in TERMINAL or alive:
                    continue
                if status.get("status") not in TERMINAL:
                    job_path = trial_dir / "job.json"
                    allocated_at = json.loads(job_path.read_text(encoding="utf-8"))["allocated_at_epoch"] if job_path.exists() else 0
                    if time.time() - allocated_at < 60:
                        active.append(None)  # short startup grace; still consumes a slot
                        continue
                    status.update(status="FAIL", trial=frozen.number, telemetry_incomplete=True,
                                  error="Interrupted/dead worker; preserved checkpoint; counts against the global cap.")
                    trial_dir.mkdir(exist_ok=True)
                    write_atomic(trial_dir / "status.json", status)
                # Reconcile the small status-write / study.tell crash window.
                state = optuna.trial.TrialState[status["status"]]
                study.tell(frozen.number, values=status.get("objective") if state.name == "COMPLETE" else None,
                           state=state, skip_if_finished=True)
            trials = study.get_trials(deepcopy=False)
            if len(trials) == cap and all(t.state.name in TERMINAL for t in trials) and not active:
                break
            stage_cap, workers, stage = scheduling_window(trials, config, preflight)
            if stage == "tpe" and not any(t.state.name == "COMPLETE" for t in trials[:config["warmup_trials"]]):
                raise RuntimeError("No completed warm-up observations: stopping for technical investigation.")
            while len(trials) < stage_cap and len(active) < workers:
                study.sampler = sampler_for_number(len(trials), config, preflight)
                trial = study.ask()  # the sole allocator; workers report/tell this trial only
                if trial.number != len(trials):
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    raise RuntimeError("Unexpected external trial allocation; refusing concurrent coordinators.")
                out = directory / f"trial_{trial.number:04d}"
                out.mkdir()
                params = preflight_parameters() if preflight else suggest_parameters(trial)
                job = {"number": trial.number, "trial_id": trial._trial_id, "stage": stage, "params": params,
                       "trial_dir": str(out), "campaign_dir": str(root), "config_path": str(config_path),
                       "study_name": study.study_name, "vm_hostname": hostname, "allocated_at_epoch": time.time()}
                write_atomic(out / "job.json", job)
                write_atomic(out / "params.json", params)
                write_atomic(out / "status.json", {"status": "ALLOCATED", "trial": trial.number, "stage": stage})
                environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(out / "matplotlib"),
                                   XDG_CACHE_HOME=str(out / "cache"), TORCH_HOME=str(out / "torch"))
                log = (out / "worker.log").open("x", encoding="utf-8")
                try:
                    child = subprocess.Popen(
                        [sys.executable, "-B", str(CODE_DIR / "optuna" / "vm_campaign_worker.py"), "--job", str(out / "job.json")],
                        cwd=out, env=environment, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                except BaseException:
                    log.close()
                    raise
                children[trial.number] = (child, log)
                write_atomic(out / "process.json", {"pid": child.pid,
                             "process_started_at": psutil.Process(child.pid).create_time()})
                active.append(child.pid)
                trials = study.get_trials(deepcopy=False)
                print(f"Allocated {stage} trial {trial.number:04d}; active processes <= {len(active)}.", flush=True)
            peak = max(peak, len(active))
            rss = 0
            for pid in active:
                if pid is not None:
                    try:
                        rss += psutil.Process(pid).memory_info().rss
                    except psutil.Error:
                        pass
            row = {"epoch": time.time(), "stage": stage, "active_workers": len(active),
                   "machine_cpu_percent": psutil.cpu_percent(), "machine_used_ram_bytes": psutil.virtual_memory().used,
                   "workers_rss_bytes": rss}
            resource_writer.writerow(row)
            resource_rows.append(row)
            time.sleep(5)
    finally:
        for child, log in children.values():
            log.close()  # workers keep their inherited handle and continue if coordinator restarts
    return peak


def preflight_checks(statuses, peak, config):
    return {
        "four_completed": len(statuses) == 4 and all(s.get("status") == "COMPLETE" for s in statuses),
        "four_distinct_processes": len({s.get("pid") for s in statuses}) == 4,
        "four_concurrent_workers": peak == 4,
        "independent_dss_circuits": len({s.get("opendss_private_circuit") for s in statuses}) == 4,
        "storage_barrier": all(s.get("storage_concurrency_barrier") for s in statuses),
        "budget_and_gradients": all(s.get("timesteps") == 2000 and s.get("gradient_updates") == 1500
                                    and s.get("replay_buffer_size") == 2000 and s.get("completed_episodes") == 11 for s in statuses),
        "save_load": all(s.get("save_load", {}).get("passed") for s in statuses),
        "same_seed_same_trajectory": len({s.get("trajectory_sha256") for s in statuses}) == 1
                                     and all(s.get("trajectory_sha256") for s in statuses),
        "single_torch_thread": all(s.get("thread_limits", {}).get("torch_intra") == 1
                                   and s.get("thread_limits", {}).get("torch_inter") == 1 for s in statuses),
        "metrics_persisted": all(s.get("metrics_verified") for s in statuses),
        "zero_incidents": all(s.get("opendss_failures") == 0 and s.get("nonfinite_incidents") == 0
                              and s.get("final_test_accesses") == 0
                              and s.get("data_access", {}).get("blocked_open_attempts") == [] for s in statuses),
    }


def bind_study(study, metadata):
    expected = metadata["campaign_id"]
    bound = study.user_attrs.get("campaign_id")
    if bound is not None and bound != expected:
        raise RuntimeError("This study belongs to another campaign/output directory.")
    if bound is None:
        if study.get_trials(deepcopy=False):
            raise RuntimeError("An existing unbound study cannot be silently adopted.")
        study.set_user_attr("campaign_id", expected)


def main():
    single_thread_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CODE_DIR / "optuna" / "vm_campaign.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--vm-hostname", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    machine = verify_vm(config, args.vm_hostname)  # before Optuna and any study creation
    import optuna
    if optuna.__version__ != "5.0.0":
        raise RuntimeError("This coordinator pins Optuna 5.0.0 for reproducible ask/tell handoff.")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with CoordinatorLock(root, config):
        frozen_packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        fingerprint = {"config_sha256": file_hash(config_path), "python": sys.version,
                       "packages": frozen_packages.strip().splitlines(),
                       "source_sha256": {p.relative_to(CODE_DIR).as_posix(): file_hash(p) for folder in ("env", "scripts", "optuna") for p in sorted((CODE_DIR / folder).glob("*.py"))},
                       "scenario_manifest_sha256": file_hash(CODE_DIR / "data" / "scenario_splits.json")}
        if fingerprint["scenario_manifest_sha256"] != config["scenario_manifest_sha256"]:
            raise RuntimeError("Scenario manifest differs from the approved baseline.")
        metadata_path = root / "campaign_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["fingerprint"] != fingerprint or metadata["machine"]["hostname"] != machine["hostname"]:
                raise RuntimeError("Resume requires the same configuration, source, scenarios and VM.")
        else:
            metadata = {"campaign_id": str(uuid.uuid4()), "created_at_epoch": time.time(), "machine": machine, "fingerprint": fingerprint, "config": config}
            write_atomic(metadata_path, metadata)
            (root / "pip_freeze.txt").write_text(frozen_packages, encoding="utf-8")
        if (root / "global_summary.json").exists():
            finished = json.loads((root / "global_summary.json").read_text(encoding="utf-8"))
            if finished.get("finished"):
                print("Campaign already completed; no additional training.", flush=True)
                return
        resource_path = root / "resources.csv"
        rows = list(csv.DictReader(resource_path.open(encoding="utf-8"))) if resource_path.exists() else []
        with resource_path.open("a", encoding="utf-8", newline="", buffering=1) as stream:
            writer = csv.DictWriter(stream, fieldnames=RESOURCE_FIELDS)
            if not rows:
                writer.writeheader()
            backend = storage(config, root)
            preflight = optuna.create_study(study_name=config["study_name"] + "_preflight",
                                           storage=backend, direction="maximize", load_if_exists=True)
            bind_study(preflight, metadata)
            if not (root / "preflight_passed.json").exists():
                peak = supervise(preflight, root, config, config_path, args.vm_hostname, writer, rows, preflight=True)
                peak = max(peak, max((int(r["active_workers"]) for r in rows if r["stage"] == "preflight"), default=0))
                statuses = [read_status(root / "preflight" / f"trial_{i:04d}") for i in range(4)]
                checks = preflight_checks(statuses, peak, config)
                passed = all(checks.values())
                write_atomic(root / ("preflight_passed.json" if passed else "preflight_failed.json"),
                             {"passed": passed, "checks": checks, "statuses": statuses})
                if not passed:
                    raise RuntimeError("Concurrent preflight failed; optimization was NOT launched.")
                print("Four-process preflight passed. Starting the approved 60-trial campaign automatically.", flush=True)
            study = optuna.create_study(study_name=config["study_name"], storage=backend, direction="maximize",
                                       pruner=make_pruner(config), load_if_exists=True)
            bind_study(study, metadata)
            supervise(study, root, config, config_path, args.vm_hostname, writer, rows)
            result = summarize(study, root, metadata, rows)
            print(f"Campaign finished: {result['counts']}. No further training is scheduled.", flush=True)


if __name__ == "__main__":
    main()

