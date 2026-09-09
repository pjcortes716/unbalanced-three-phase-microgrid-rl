"""VM campaign policy and lazy storage adapters; importing starts no study."""
from __future__ import annotations

# Permit both direct CLI execution and package imports from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import inspect
import json
import math
from numbers import Real
import os
from pathlib import Path
import socket

CODE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINTS = (16800, 33600, 50400, 67200, 84000, 100800)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
ARCHITECTURES = {"128x128": (128, 128), "256x256": (256, 256), "256x256x256": (256, 256, 256)}
CATEGORICALS = {
    "net_arch": tuple(ARCHITECTURES), "batch_size": (128, 256, 512),
    "gamma": (0.99, 0.995, 0.999), "learning_starts": (1680, 3360, 8400),
    "ent_coef": ("auto_0.01", "auto_0.1", "auto_1.0"),
}


def single_thread_environment():
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"total_trials": 60, "warmup_trials": 12, "warmup_workers": 6,
                "tpe_trials": 48, "tpe_workers": 28, "max_timesteps": 100800,
                "preflight_workers": 4, "preflight_timesteps": 2000, "preflight_learning_starts": 500,
                "c_unserved": 1.0, "lambda_pf": 1.0, "extreme_events": None,
                "training_split": "train", "evaluation_split": "optuna_validation"}
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Approved campaign requires {key}={value!r}.")
    if config["checkpoints"] != list(CHECKPOINTS):
        raise ValueError("Unexpected validation checkpoints.")
    if config["prune_from_step"] not in (33600, 50400):
        raise ValueError("Pruning may start only after the second or third full validation.")
    if config["storage"] not in ("journal", "postgresql"):
        raise ValueError("Only local JournalStorage or PostgreSQL is allowed; never SQLite.")
    if not isinstance(config["learning_seed"], int) or config["learning_seed"] < 0:
        raise ValueError("Invalid training seed.")
    validation_pairs(tuple(f"week_{i}" for i in range(6)), config)
    return config


def verify_vm(config, expected_hostname):
    """Fail before importing Optuna/Torch or launching any local optimization."""
    if config["blocked_local_hostname"] == "SET_LOCAL_HOSTNAME_BEFORE_CAMPAIGN":
        raise RuntimeError("Set blocked_local_hostname to your local workstation before deploying the campaign.")
    hostname = socket.gethostname()
    if hostname.casefold() == config["blocked_local_hostname"].casefold():
        raise RuntimeError("The local PC is explicitly blocked from this Optuna campaign.")
    if hostname.casefold() != expected_hostname.casefold():
        raise RuntimeError(f"VM identity mismatch: expected {expected_hostname!r}, got {hostname!r}.")
    import psutil
    cpus = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), "cpu_affinity") else os.cpu_count()
    if cpus < 30 or psutil.virtual_memory().total < 110_000_000_000:
        raise RuntimeError("Expected VM allocation: at least 30 available CPUs and approximately 124 GB RAM.")
    return {"hostname": hostname, "available_cpus": cpus, "ram_bytes": psutil.virtual_memory().total}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def storage(config, campaign_dir):
    import optuna
    if config["storage"] == "postgresql":
        url = os.environ.get("OPTUNA_STORAGE_URL", "")
        if not url.startswith(("postgresql://", "postgresql+psycopg://", "postgresql+psycopg2://")):
            raise ValueError("Set OPTUNA_STORAGE_URL to a PostgreSQL URL; it is never written to config/logs.")
        # Coordinator/worker PID supervision implements recovery for ask/tell.
        # Automatic RDB heartbeat is deliberately NOT claimed for ask/tell.
        return optuna.storages.RDBStorage(
            url, engine_kwargs={"pool_size": 1, "max_overflow": 1, "pool_pre_ping": True})
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    path = Path(campaign_dir).resolve() / "optuna_journal.log"
    import psutil
    for mount in sorted(psutil.disk_partitions(all=True), key=lambda p: len(p.mountpoint), reverse=True):
        if path.is_relative_to(Path(mount.mountpoint).resolve()):
            if mount.fstype.lower() in {"nfs", "nfs4", "cifs", "smbfs", "fuse.sshfs"}:
                raise RuntimeError("JournalStorage must reside on the VM's local disk, not a network mount.")
            break
    return optuna.storages.JournalStorage(JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path))))


def make_pruner(config):
    import optuna
    return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=config["prune_from_step"],
                                       interval_steps=16800, n_min_trials=5)


def sampler_for_number(number, config, preflight=False):
    import optuna
    seed = config["sampler_seed"] + number
    if preflight or number < config["warmup_trials"]:
        return optuna.samplers.RandomSampler(seed=seed)
    options = {"seed": seed, "n_startup_trials": 0}
    if "constant_liar" in inspect.signature(optuna.samplers.TPESampler).parameters:
        options["constant_liar"] = True
    else:
        raise RuntimeError("Pinned Optuna must support constant_liar=True.")
    return optuna.samplers.TPESampler(**options)


def suggest_parameters(trial):
    params = {name: trial.suggest_categorical(name, choices) for name, choices in CATEGORICALS.items()}
    params["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True)
    params["tau"] = trial.suggest_float("tau", 0.002, 0.02, log=True)
    return params


def preflight_parameters():
    return {"net_arch": "256x256", "batch_size": 256, "gamma": 0.99, "learning_starts": 500,
            "ent_coef": "auto", "learning_rate": 3e-4, "tau": 0.005}


def can_prune(number, timestep, config):
    return number >= config["warmup_trials"] and timestep >= config["prune_from_step"] and timestep < config["max_timesteps"]


def validation_pairs(reference_ids, config):
    if len(reference_ids) != 6 or len(set(reference_ids)) != 6:
        raise ValueError("Exactly six validation base scenarios are required.")
    first, second = config["validation_seed_sets"]
    if (len(first) != 6 or len(second) != 6
            or any(not isinstance(seed, int) or seed < 0 or seed >= 2**32 for seed in first + second)
            or any(a == b for a, b in zip(first, second))):
        raise ValueError("Exactly two realization seeds per validation week are required.")
    return tuple((scenario, seed) for i, scenario in enumerate(reference_ids) for seed in (first[i], second[i]))


def assert_campaign_science(experiment):
    for key, value in {"c_unserved": 1.0, "lambda_pf": 1.0, "extreme_events": None,
                       "training_split": "train", "evaluation_split": "optuna_validation",
                       "buffer_size": 100000, "train_freq": 1, "gradient_steps": 1,
                       "target_entropy": "auto", "policy": "MlpPolicy"}.items():
        if getattr(experiment, key) != value:
            raise ValueError(f"Scientific/fixed parameter changed: {key}.")



def assert_metrics_finite(value, path="metrics"):
    """Validate numeric metric leaves before any CSV write; None denotes unavailable."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert_metrics_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_metrics_finite(item, f"{path}[{index}]")
    elif hasattr(value, "tolist"):
        assert_metrics_finite(value.tolist(), path)
    elif isinstance(value, Real) and not math.isfinite(value):
        raise RuntimeError(f"NaN/Inf in {path}.")
