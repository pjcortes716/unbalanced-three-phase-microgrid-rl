"""Reusable, guarded baseline SAC experiments for benchmark_v1.

No Optuna implementation and no final_test policy access. SAC updates are
delegated unchanged to Stable-Baselines3 2.7.1; checks only observe or abort.
"""
from __future__ import annotations

# Permit both direct CLI execution and package imports from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import weakref

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
import torch

from env.Ambiente_industrial import MicrorredIndustrial
from env.scenario_catalog import ScenarioCatalog

CODE_DIR = Path(__file__).resolve().parents[1]
ALLOWED_TUNING_SPLITS = ("train", "optuna_validation")
HORIZON = 168
VALIDATION_SEEDS = (420001, 420002, 420003, 420004, 420005, 420006)
TRAJECTORY_RTOL = 1e-7
TRAJECTORY_ATOL = 1e-7


class ExperimentSanityError(RuntimeError):
    pass


def validate_tuning_split(split: str) -> str:
    if split not in ALLOWED_TUNING_SPLITS:
        raise ValueError(f"Tuning controller forbids split={split!r}; only {ALLOWED_TUNING_SPLITS} are allowed.")
    return split


@dataclass(frozen=True)
class SACExperimentConfig:
    seed: int = 42
    policy: str = "MlpPolicy"
    net_arch: tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    buffer_size: int = 100000
    learning_starts: int = 3360
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    target_entropy: str = "auto"
    total_timesteps: int = 20160
    training_split: str = "train"
    evaluation_split: str = "optuna_validation"
    lambda_pf: float = 1.0
    c_unserved: float = 1.0
    extreme_events: None = None
    device: str = "cpu"
    torch_threads: int = 1
    validation_seeds: tuple[int, ...] = VALIDATION_SEEDS
    validation_scenario_count: int = 6
    allow_partial_last_episode: bool = False
    purpose: str = "smoke_pipeline"

    def validate(self):
        validate_tuning_split(self.training_split)
        validate_tuning_split(self.evaluation_split)
        if self.training_split != "train" or self.evaluation_split != "optuna_validation":
            raise ValueError("Training must use train and evaluation must use optuna_validation.")
        if self.extreme_events is not None:
            raise ValueError("Initial SAC studies require extreme_events=None.")
        if self.policy != "MlpPolicy" or self.device != "cpu":
            raise ValueError("This reproducibility protocol uses MlpPolicy on CPU.")
        for name in ("seed", "learning_starts"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        for name in ("buffer_size", "batch_size", "total_timesteps", "torch_threads", "train_freq", "gradient_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.train_freq != 1 or self.gradient_steps != 1:
            raise ValueError("The guarded controller currently checks every single-step SAC update.")
        if self.total_timesteps % HORIZON and not self.allow_partial_last_episode:
            raise ValueError("Partial final episodes require explicit allow_partial_last_episode=True.")
        if self.validation_scenario_count not in (2, 6):
            raise ValueError("Use two reference scenarios for the reduced technical check, or six for the full protocol.")
        if len(self.validation_seeds) != 6 or any(not isinstance(s, int) or s < 0 for s in self.validation_seeds):
            raise ValueError("Exactly six fixed nonnegative realization seeds are required.")
        if not self.net_arch or any(not isinstance(n, int) or n <= 0 for n in self.net_arch):
            raise ValueError("net_arch must contain positive layer sizes.")
        for name in ("learning_rate", "c_unserved", "lambda_pf", "gamma", "tau"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.gamma > 1 or self.tau > 1:
            raise ValueError("gamma and tau must not exceed one.")

    @classmethod
    def from_dict(cls, values: dict):
        values = dict(values)
        for name in ("net_arch", "validation_seeds"):
            if name in values:
                values[name] = tuple(values[name])
        result = cls(**values)
        result.validate()
        return result


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                               default=json_value) + "\n", encoding="utf-8")


def assert_finite(name: str, values):
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise ExperimentSanityError(f"NaN/Inf detected in {name}.")


class CsvSink:
    """Exclusive CSV creation; nested metrics are preserved as JSON cells."""
    def __init__(self, path: Path, fields=None):
        self.path = path
        self.file = path.open("x", encoding="utf-8", newline="")
        self.fields = fields
        self.writer = None
        self.rows = []
        if fields:
            self.writer = csv.DictWriter(self.file, fieldnames=fields)
            self.writer.writeheader()

    def write(self, row, *, retain=True, flush=True):
        if self.writer is None:
            self.fields = list(row)
            self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
            self.writer.writeheader()
        encoded = {k: json.dumps(v, ensure_ascii=False, allow_nan=False, default=json_value)
                   if isinstance(v, (dict, list, tuple, np.ndarray)) else v for k, v in row.items()}
        self.writer.writerow(encoded)
        if retain:
            self.rows.append(dict(row))
        if flush:
            self.file.flush()

    def close(self):
        self.file.close()


class DataAccessAudit:
    """Audit actual Python file opens and block forbidden data BEFORE the open.

    Hooks cannot be removed by Python; after context exit this hook is inactive.
    The single-process controller does not run environments concurrently.
    """
    def __init__(self, path: Path):
        self.path = path
        self.active = False
        self.stage = "setup"
        self.records = []
        self.blocked = []
        self.file = None

    def __enter__(self):
        self.file = self.path.open("x", encoding="utf-8")
        self.active = True
        sys.addaudithook(self._hook)
        return self

    def _hook(self, event, args):
        if not self.active or event != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        raw = os.fsdecode(args[0])
        path = Path(raw).resolve()
        data_root = (CODE_DIR / "data").resolve()
        if not path.is_relative_to(data_root):
            return
        pieces = path.relative_to(data_root).parts
        split = pieces[0].lower() if pieces else ""
        if path.suffix.lower() != ".npz":
            return
        allowed = split in ALLOWED_TUNING_SPLITS
        if self.stage == "training":
            allowed = allowed and split == "train"
        elif self.stage.startswith("validation"):
            allowed = allowed and split == "optuna_validation"
        record = {"stage": self.stage, "split": split, "path": str(path), "allowed": allowed,
                  "operation": "open_attempt"}
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()
        (self.records if allowed else self.blocked).append(record)
        if not allowed:
            raise ExperimentSanityError(f"Forbidden dataset access before open: {split}, stage={self.stage}, path={path}")

    @contextmanager
    def phase(self, name):
        previous = self.stage
        self.stage = name
        try:
            yield
        finally:
            self.stage = previous

    def summary(self):
        return {
            "allowed_open_count": len(self.records),
            "allowed_splits": sorted({r["split"] for r in self.records}),
            "allowed_paths": sorted({r["path"] for r in self.records}),
            "blocked_open_attempts": self.blocked,
            "final_test_open_count": sum(r["split"] == "final_test" for r in self.records),
        }

    def __exit__(self, *args):
        self.active = False
        self.file.close()


def episode_seed(learning_seed: int, episode_index: int) -> int:
    return int(np.random.SeedSequence([learning_seed, 1001, episode_index]).generate_state(1, dtype=np.uint32)[0])


class EpisodeGuard(gym.Wrapper):
    """Seed each episode explicitly and validate every transition before replay."""
    def __init__(self, env, config, split, sink=None, events_file=None, training=False):
        super().__init__(env)
        self.config = config
        self.expected_split = validate_tuning_split(split)
        self.training = training
        self.sink = sink
        self.events_file = events_file
        self.started_episodes = 0
        self.completed_episodes = 0
        self.total_steps = 0
        self.episode_steps = 0
        self.realization_seed = None
        self.scenario_id = None
        self.active_episode = False
        self.last_metrics = None
        self.solver_failures = 0
        self.reward_checks = 0
        self.observation_checks = 0
        self.action_checks = 0
        self.base_week_ids = None

    def _event(self, record):
        if self.events_file:
            self.events_file.write(json.dumps(record, default=json_value, allow_nan=False) + "\n")
            self.events_file.flush()

    def reset(self, *, seed=None, options=None):
        if self.active_episode and self.episode_steps:
            raise ExperimentSanityError("Reset would silently discard an incomplete episode.")
        if self.training:
            if seed not in (None, self.config.seed):
                raise ExperimentSanityError("Unexpected training master seed.")
            if options:
                raise ExperimentSanityError("Training scenarios must be sampled from train.")
            realization = episode_seed(self.config.seed, self.started_episodes)
        else:
            if seed is None or not options or "scenario_id" not in options:
                raise ExperimentSanityError("Evaluation requires an explicit realization seed and scenario_id.")
            realization = int(seed)
        obs, info = self.env.reset(seed=realization, options=options)
        assert_finite("reset observation", obs)
        self.observation_checks += 1
        if info["split"] != self.expected_split:
            raise ExperimentSanityError("Reset returned a scenario from another split.")
        self.env.scenario_catalog.resolve(self.expected_split, info["scenario_id"])
        self.realization_seed = realization
        self.scenario_id = info["scenario_id"]
        self.base_week_ids = info["base_week_ids"]
        self.episode_steps = 0
        self.active_episode = True
        self.started_episodes += 1
        info["realization_seed"] = realization
        info["learning_seed"] = self.config.seed
        self._event({"event": "reset", "split": self.expected_split, "scenario_id": self.scenario_id,
                     "episode_index": self.started_episodes - 1, "realization_seed": realization,
                     "learning_seed": self.config.seed, "base_week_ids": self.base_week_ids})
        return obs, info

    def step(self, action):
        if not self.active_episode:
            raise ExperimentSanityError("Step outside an active episode.")
        assert_finite("policy action", action)
        self.action_checks += 1
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except Exception as error:
            self.solver_failures = self.env.get_episode_metrics()["failed_solver_attempts"]
            self._event({"event": "failed_transition", "t": self.episode_steps,
                         "scenario_id": self.scenario_id, "realization_seed": self.realization_seed,
                         "error": repr(error), "solver_failures": self.solver_failures})
            raise
        assert_finite("step observation", obs)
        assert_finite("step reward", reward)
        self.observation_checks += 1
        if info["split"] != self.expected_split or info["scenario_id"] != self.scenario_id:
            raise ExperimentSanityError("Scenario/split changed during an episode.")
        if not info["electrical_converged"]:
            self.solver_failures += 1
            raise ExperimentSanityError("OpenDSS returned nonconvergence.")
        costs = [info[k] for k in ("energy_cost_usd", "pf_penalty_usd", "truck_departure_penalty_usd")]
        assert_finite("reward components", costs)
        if not math.isclose(float(reward), -sum(costs), rel_tol=1e-10, abs_tol=1e-8):
            raise ExperimentSanityError(f"Reward decomposition mismatch at t={info['t']}.")
        if info["c_unserved_usd_per_kwh"] != self.config.c_unserved:
            raise ExperimentSanityError("Scientific coefficient changed.")
        self.reward_checks += 1
        self.episode_steps += 1
        self.total_steps += 1
        if info["t"] != self.episode_steps - 1 or info["next_t"] != self.episode_steps:
            raise ExperimentSanityError("Temporal contract mismatch.")
        if truncated or bool(terminated) != (self.episode_steps == HORIZON):
            raise ExperimentSanityError(f"Unexpected episode length/termination: {self.episode_steps}.")
        if terminated:
            self.active_episode = False
            self.completed_episodes += 1
            metrics = dict(info["episode_metrics"])
            if metrics["steps"] != HORIZON or not metrics["complete"]:
                raise ExperimentSanityError("Incomplete terminal episode metrics.")
            if metrics["failed_solver_attempts"]:
                raise ExperimentSanityError("Episode contains a failed OpenDSS attempt.")
            self.last_metrics = {
                "episode": self.completed_episodes, "environment_steps": self.total_steps,
                "learning_seed": self.config.seed, "realization_seed": self.realization_seed,
                "base_week_ids": self.base_week_ids, **metrics,
                "voltage_violation_rate": metrics["voltage_violation_steps"] / HORIZON,
                "pf_violation_rate": metrics["pf_violation_steps"] / HORIZON,
                "environment_step_seconds": info["episode_runtime"]["step_seconds_total"],
            }
            if self.sink:
                self.sink.write(self.last_metrics)
            self._event({"event": "episode_complete", "episode": self.completed_episodes,
                         "environment_steps": self.total_steps, "scenario_id": self.scenario_id,
                         "realization_seed": self.realization_seed, "return_total": metrics["return_total"]})
        return obs, reward, terminated, truncated, info


def make_environment(config: SACExperimentConfig, split: str, *, sink=None, events_file=None, training=False):
    validate_tuning_split(split)  # before environment construction or any data access
    if training and split != "train":
        raise ValueError("Training is restricted to train.")
    if not training and split != "optuna_validation":
        raise ValueError("Policy evaluation is restricted to optuna_validation.")
    env = MicrorredIndustrial(split=split, c_unserved=config.c_unserved,
                              lambda_pf=config.lambda_pf, extreme_events=None)
    return EpisodeGuard(env, config, split, sink=sink, events_file=events_file, training=training)


def assert_model_finite(model):
    with torch.no_grad():
        for name, tensor in model.policy.state_dict().items():
            if not bool(torch.isfinite(tensor).all()):
                raise ExperimentSanityError(f"NaN/Inf in SAC parameter/buffer {name}.")
        for name, parameter in model.policy.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise ExperimentSanityError(f"NaN/Inf in SAC gradient {name}.")
        if model.log_ent_coef is not None:
            if not bool(torch.isfinite(model.log_ent_coef).all()) or not bool(torch.isfinite(model.log_ent_coef.exp()).all()):
                raise ExperimentSanityError("NaN/Inf in SAC entropy coefficient.")


TRAINING_FIELDS = ["environment_steps", "gradient_updates", "elapsed_seconds", "learning_rate",
                   "actor_loss", "critic_loss", "ent_coef", "ent_coef_loss", "finite_check_passed"]


class CheckedSAC(SAC):
    """Unchanged SB3 SAC optimization plus checks after every gradient update."""
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_training_sink", "_training_started_at"]

    def train(self, gradient_steps, batch_size=64):
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)
        values = self.logger.name_to_value
        for key in ("train/actor_loss", "train/critic_loss", "train/ent_coef", "train/ent_coef_loss"):
            if key in values:
                assert_finite(key, values[key])
        assert_model_finite(self)
        self._finite_update_checks = getattr(self, "_finite_update_checks", 0) + 1
        sink = getattr(self, "_training_sink", None)
        if sink is not None:
            row = {"environment_steps": self.num_timesteps, "gradient_updates": self._n_updates,
                   "elapsed_seconds": time.perf_counter() - self._training_started_at,
                   "learning_rate": values["train/learning_rate"],
                   "actor_loss": values["train/actor_loss"], "critic_loss": values["train/critic_loss"],
                   "ent_coef": values["train/ent_coef"], "ent_coef_loss": values.get("train/ent_coef_loss"),
                   "finite_check_passed": True}
            sink.write(row, retain=False, flush=self._n_updates % HORIZON == 0)


class ProgressCallback(BaseCallback):
    def _on_step(self):
        assert_finite("SB3 replay actions", self.locals["actions"])
        assert_finite("SB3 vector reward", self.locals["rewards"])
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            metrics = info["episode_metrics"]
            episode = self.num_timesteps // HORIZON
            if episode % 5 == 0:
                print(json.dumps({"progress": "training", "timesteps": self.num_timesteps,
                                  "episodes": episode, "return": metrics["return_total"]}), flush=True)
        return True


def build_model(config, env):
    if stable_baselines3.__version__ != "2.7.1":
        raise RuntimeError(f"Expected SB3 2.7.1, found {stable_baselines3.__version__}.")
    model = CheckedSAC(
        config.policy, env, policy_kwargs={"net_arch": list(config.net_arch)},
        learning_rate=config.learning_rate, buffer_size=config.buffer_size,
        learning_starts=config.learning_starts, batch_size=config.batch_size,
        gamma=config.gamma, tau=config.tau, train_freq=config.train_freq,
        gradient_steps=config.gradient_steps, ent_coef=config.ent_coef,
        target_entropy=config.target_entropy, seed=config.seed, device=config.device, verbose=0)
    model.set_logger(Logger(folder=None, output_formats=[]))
    assert_model_finite(model)
    return model


def train_model(model, config, training_sink):
    """Reusable training entry point; no CLI logic or checkpoint selection."""
    model._training_sink = training_sink
    model._training_started_at = time.perf_counter()
    model.learn(total_timesteps=config.total_timesteps, callback=ProgressCallback(), log_interval=1)
    elapsed = time.perf_counter() - model._training_started_at
    training_sink.file.flush()
    assert_model_finite(model)
    if model.num_timesteps != config.total_timesteps:
        raise ExperimentSanityError("Training stopped before the configured timestep budget.")
    return {"training_wall_seconds": elapsed, "timesteps": model.num_timesteps,
            "gradient_updates": model._n_updates,
            "finite_update_checks": getattr(model, "_finite_update_checks", 0),
            "replay_buffer_size": model.replay_buffer.size()}


def validation_protocol(config):
    validate_tuning_split(config.evaluation_split)
    if config.evaluation_split != "optuna_validation":
        raise ValueError("Main evaluation protocol requires optuna_validation.")
    ids = ScenarioCatalog().reference_scenario_ids("optuna_validation")
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ExperimentSanityError("Validation protocol must contain exactly six unique reference scenarios.")
    return tuple(zip(ids, config.validation_seeds))[:config.validation_scenario_count]


def evaluate_scenarios(model, config, protocol, *, sink=None, events_file=None, capture_first=False):
    """Deterministic validation only, also used for the one-scenario load check."""
    allowed_ids = {scenario for scenario, _ in validation_protocol(config)}
    protocol = tuple(protocol)
    if not protocol or any(scenario not in allowed_ids for scenario, _ in protocol):
        raise ValueError("Evaluation is restricted to known optuna_validation reference scenarios.")
    env = make_environment(config, "optuna_validation", sink=sink, events_file=events_file)
    rows, first_trace = [], None
    try:
        for index, (scenario_id, realization_seed) in enumerate(protocol):
            obs, _ = env.reset(seed=realization_seed, options={"scenario_id": scenario_id})
            trace = {"observations": [obs.copy()], "actions": [], "rewards": [],
                     "p_grid_kw": [], "voltages_pu": [], "pf_phase": [], "soc": [],
                     "costs": [], "losses_kw": []}
            for t in range(HORIZON):
                action, _ = model.predict(obs, deterministic=True)
                assert_finite("deterministic evaluation action", action)
                obs, reward, terminated, truncated, info = env.step(action)
                if capture_first and index == 0:
                    trace["observations"].append(obs.copy())
                    trace["actions"].append(action.copy())
                    trace["rewards"].append(reward)
                    trace["p_grid_kw"].append(info["p_grid_kw"])
                    trace["voltages_pu"].append([info["voltages_pu"][f"bus{i}"] for i in range(2, 7)])
                    trace["pf_phase"].append(info["pf_phase"])
                    trace["soc"].append([info["controls"][name]["soc_after"] for name in ("bess", "truck5", "truck6")])
                    trace["costs"].append([info[name] for name in ("energy_cost_usd", "pf_penalty_usd", "truck_departure_penalty_usd")])
                    trace["losses_kw"].append(info["p_losses_kw"])
            if not terminated or truncated:
                raise ExperimentSanityError("Validation failed to finish its complete week.")
            rows.append(dict(env.last_metrics))
            if capture_first and index == 0:
                first_trace = {key: np.asarray(value) for key, value in trace.items()}
    finally:
        env.close()
    return rows, first_trace


def evaluate_validation(model, config, **kwargs):
    return evaluate_scenarios(model, config, validation_protocol(config), **kwargs)


def compare_trajectories(before, after):
    if before.keys() != after.keys():
        raise ExperimentSanityError("Save/load trajectory fields differ.")
    differences = {}
    for key in before:
        a, b = np.asarray(before[key]), np.asarray(after[key])
        assert_finite(f"save/load {key} before", a)
        assert_finite(f"save/load {key} after", b)
        if a.shape != b.shape or not np.allclose(a, b, rtol=TRAJECTORY_RTOL, atol=TRAJECTORY_ATOL):
            raise ExperimentSanityError(f"Save/load deterministic trajectory mismatch in {key}.")
        differences[key] = float(np.max(np.abs(a - b)))
    return {"passed": True, "rtol": TRAJECTORY_RTOL, "atol": TRAJECTORY_ATOL,
            "maximum_absolute_differences": differences}


def code_and_environment_manifest(config):
    catalog = ScenarioCatalog()  # metadata only; no final_test NPZ inspection
    names = ('env/Ambiente_industrial.py', 'env/battery.py', 'env/benchmark_metrics.py', 'env/extreme_events.py', 'env/oficinas.py', 'env/opendss_easy.py', 'env/price_sampler.py', 'env/PV_plant.py', 'env/scenario_catalog.py', 'env/v2g_bus_5.py', 'env/vehiculos_Zoe.py', 'scripts/sac_experiment.py', 'scripts/train_sac.py', 'scripts/evaluate_sac.py', 'scripts/sac_report.py')
    git = {}
    try:
        git["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_DIR, text=True).strip()
        git["status_porcelain"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=CODE_DIR, text=True).splitlines()
        git["dirty"] = bool(git["status_porcelain"])
    except (OSError, subprocess.CalledProcessError) as error:
        git["unavailable_reason"] = str(error)
    return {
        "benchmark": "benchmark_v1", "scenario_manifest_sha256": catalog.manifest_sha256,
        "git": git, "code_sha256": {name: hashlib.sha256((CODE_DIR / name).read_bytes()).hexdigest() for name in names},
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("stable-baselines3", "torch", "gymnasium", "numpy",
                     "pandas", "opendssdirect.py", "dss-python", "dss-python-backend")},
        "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(), "device": config.device,
        "torch_threads": config.torch_threads, "torch_deterministic_algorithms": True,
        "splits_available_to_controller": {
            split: catalog.manifest["splits"][split] for split in ALLOWED_TUNING_SPLITS},
        "final_test_policy": "Forbidden to this controller, including evaluation CLI; no policy file opens.",
        "forecast_model": "synthetic noisy forecasts", "observation_dim": 114, "action_dim": 12,
        "horizon": HORIZON, "c_unserved": config.c_unserved, "lambda_pf": config.lambda_pf,
        "extreme_events": None,
    }


def aggregate_episode_rows(rows):
    keys = ("return_total", "energy_cost_usd", "pf_penalty_usd", "truck_departure_penalty_usd",
            "truck_energy_deficit_kwh", "truck_departure_count", "truck_departure_violation_count",
            "voltage_violation_steps", "pf_violation_steps", "import_kwh", "export_kwh", "losses_kwh",
            "pv_available_kwh", "pv_utilized_kwh", "curtailment_kwh", "bess_soc_final",
            "vuf_max_pct", "transformer_loading_max_pct")
    if not rows:
        return {"episodes": 0}
    result = {"episodes": len(rows), "means": {key: float(np.mean([r[key] for r in rows])) for key in keys},
              "totals": {key: float(sum(r[key] for r in rows)) for key in keys}}
    steps = sum(r["steps"] for r in rows)
    result.update({
        "steps": steps, "voltage_violation_rate": sum(r["voltage_violation_steps"] for r in rows) / steps,
        "pf_violation_rate": sum(r["pf_violation_steps"] for r in rows) / steps,
        "voltage_violation_max_pu": max(r["voltage_violation_max_pu"] for r in rows),
        "vuf_max_pct": max(r["vuf_max_pct"] for r in rows),
        "transformer_loading_max_pct": max(r["transformer_loading_max_pct"] for r in rows),
        "pv_utilization_fraction": (sum(r["pv_utilized_kwh"] for r in rows) /
                                   sum(r["pv_available_kwh"] for r in rows)),
        "failed_solver_attempts": sum(r["failed_solver_attempts"] for r in rows),
    })
    return result


def run_experiment(config: SACExperimentConfig, output_dir: str | Path):
    """Full smoke pipeline; creates a NEW run and never overwrites an earlier run."""
    config.validate()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sinks, env, events = [], None, None
    audit = DataAccessAudit(output_dir / "data_access.jsonl")
    summary = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "seed": config.seed, "output_dir": str(output_dir)}
    write_json(output_dir / "run_summary.json", summary)
    try:
        with audit:
            torch.set_num_threads(config.torch_threads)
            torch.use_deterministic_algorithms(True)
            manifest = code_and_environment_manifest(config)
            protocol = validation_protocol(config)
            write_json(output_dir / "config.json", {
                "experiment": asdict(config), "algorithm": "baseline SAC",
                "validation_protocol": [{"scenario_id": s, "realization_seed": seed} for s, seed in protocol],
                "training_seed_recipe": "uint32(SeedSequence([learning_seed,1001,zero_based_episode_index]))",
                "reward_formula": "-energy_cost_usd-pf_penalty_usd-truck_departure_penalty_usd",
                "checkpoint_selection": "none; final model only; validation is descriptive",
                "purpose": config.purpose,
                "interpretation": "technical pipeline verification only; no policy-quality conclusion",
            })
            write_json(output_dir / "environment_manifest.json", manifest)
            snapshot = output_dir / "code_snapshot"
            snapshot.mkdir()
            for name in manifest["code_sha256"]:
                (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
                (snapshot / name).write_bytes((CODE_DIR / name).read_bytes())
            events = (output_dir / "episode_events.jsonl").open("x", encoding="utf-8")
            episode_sink = CsvSink(output_dir / "episode_metrics.csv")
            training_sink = CsvSink(output_dir / "training_log.csv", TRAINING_FIELDS)
            initial_sink = CsvSink(output_dir / "validation_initial.csv")
            final_sink = CsvSink(output_dir / "validation_final.csv")
            reload_sink = CsvSink(output_dir / "validation_reloaded.csv")
            sinks.extend((episode_sink, training_sink, initial_sink, final_sink, reload_sink))
            with audit.phase("training"):
                env = make_environment(config, "train", sink=episode_sink, events_file=events, training=True)
                model = build_model(config, env)
            with audit.phase("validation_initial"):
                initial_rows, _ = evaluate_validation(model, config, sink=initial_sink, events_file=events)
            print(json.dumps({"progress": "initial_validation_complete", "episodes": len(initial_rows)}), flush=True)
            with audit.phase("training"):
                train_stats = train_model(model, config, training_sink)
            if env.completed_episodes != config.total_timesteps // HORIZON:
                raise ExperimentSanityError("Wrong number of completed training episodes.")
            partial = None
            remainder = config.total_timesteps % HORIZON
            if remainder:
                partial = env.unwrapped.get_episode_metrics(include_runtime=False)
                if not config.allow_partial_last_episode or partial["steps"] != remainder or partial["complete"]:
                    raise ExperimentSanityError("Unexplained incomplete final episode.")
                partial.update({"realization_seed": env.realization_seed,
                                "learning_seed": config.seed,
                                "stop_reason": "explicit technical timestep budget; no environment truncation injected"})
                write_json(output_dir / "partial_episode_metrics.json", partial)
                env._event({"event": "partial_episode_at_budget", "steps": remainder,
                            "scenario_id": env.scenario_id, "realization_seed": env.realization_seed,
                            "reason": partial["stop_reason"]})
            with audit.phase("validation_final"):
                final_rows, reference_trace = evaluate_validation(
                    model, config, sink=final_sink, events_file=events, capture_first=True)
            model_path = output_dir / "model_final.zip"
            model.save(model_path)
            if not model_path.is_file() or model_path.stat().st_size == 0:
                raise ExperimentSanityError("Model save did not produce an archive.")
            np.savez_compressed(output_dir / "trajectory_validation_final_01.npz", **reference_trace)
            model_reference = weakref.ref(model)
            del model
            gc.collect()
            if model_reference() is not None:
                raise ExperimentSanityError("Original SAC instance remained alive before load.")
            loaded = CheckedSAC.load(model_path, device=config.device)
            assert_model_finite(loaded)
            with audit.phase("validation_reload"):
                reloaded_rows, reloaded_trace = evaluate_scenarios(
                    loaded, config, protocol[:1], sink=reload_sink, events_file=events, capture_first=True)
            roundtrip = compare_trajectories(reference_trace, reloaded_trace)
            roundtrip.update({"original_model_destroyed": True, "scenario_id": protocol[0][0],
                              "realization_seed": protocol[0][1], "steps_compared": HORIZON})
            np.savez_compressed(output_dir / "trajectory_validation_reloaded_01.npz", **reloaded_trace)
            del loaded
            gc.collect()
            access = audit.summary()
            if access["final_test_open_count"] or access["blocked_open_attempts"]:
                raise ExperimentSanityError("Dataset access guard detected a forbidden attempt.")
            train_rows = episode_sink.rows
            if {row["split"] for row in train_rows} != {"train"}:
                raise ExperimentSanityError("Non-train episode in training artifacts.")
            if {row["split"] for row in initial_rows + final_rows + reloaded_rows} != {"optuna_validation"}:
                raise ExperimentSanityError("Non-validation episode in evaluation artifacts.")
            summary.update({
                "status": "completed", **train_stats,
                "wall_time_seconds": time.perf_counter() - started,
                "wall_time_definition": "pipeline through initial/final validation and save/load; excludes report rendering",
                "completed_training_episodes": env.completed_episodes,
                "partial_episode_steps": remainder,
                "partial_episode_recorded": partial is not None,
                "training_all": aggregate_episode_rows(train_rows),
                "training_last20": aggregate_episode_rows(train_rows[-20:]),
                "validation_initial": aggregate_episode_rows(initial_rows),
                "validation_final": aggregate_episode_rows(final_rows),
                "save_load": roundtrip, "data_access": access,
                "sanity": {"nonfinite_incidents": 0, "reward_checks_training": env.reward_checks,
                           "observation_checks_training": env.observation_checks, "action_checks_training": env.action_checks,
                           "opendss_failures_training": env.solver_failures,
                           "opendss_failures_total": sum(row["failed_solver_attempts"] for row in
                                train_rows + initial_rows + final_rows + reloaded_rows)},
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "target_entropy_resolved": -12.0,
                "final_test_used": False, "optuna_started": False,
                "performance_interpretation": "descriptive only; no hyperparameter or checkpoint selection",
                "warnings": [
                    "Installed legacy gym emits an import warning; environments and SAC use Gymnasium.",
                    "The benchmark observation Box retains infinite bounds; every actual observation is checked finite.",
                    (f"The explicit timestep budget leaves a partial episode of {remainder} steps; recorded separately."
                     if remainder else "SB3 auto-resets after the final completed episode and takes no further action."),
                ],
            })
            write_json(output_dir / "run_summary.json", summary)
    except BaseException as error:
        summary.update({"status": "failed", "wall_time_seconds": time.perf_counter() - started,
                        "error": repr(error), "traceback": traceback.format_exc(),
                        "data_access": audit.summary(),
                        "completed_training_episodes": 0 if env is None else env.completed_episodes,
                        "timesteps": 0 if env is None else env.total_steps,
                        "opendss_failures": 0 if env is None else env.solver_failures})
        write_json(output_dir / "run_summary.json", summary)
        (output_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if env is not None:
            env.close()
        for sink in sinks:
            sink.close()
        if events is not None:
            events.close()
    from scripts.sac_report import create_report
    create_report(output_dir)
    print(json.dumps({"progress": "completed", "timesteps": summary["timesteps"],
                      "episodes": summary["completed_training_episodes"],
                      "wall_time_seconds": summary["wall_time_seconds"], "save_load": summary["save_load"]["passed"]}), flush=True)
    return summary

