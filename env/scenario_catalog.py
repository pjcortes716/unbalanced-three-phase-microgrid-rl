"""Immutable-by-convention benchmark_v1 split catalog and scenario identifiers.

Only the selected split's NPZ files are opened by the environment. The manifest
contains provenance, but never makes final-test timeseries part of train samplers.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

SPLIT_NAMES = ("train", "optuna_validation", "final_test")
SOURCE_NAMES = ("office", "wh1", "wh2", "solar")
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "data" / "scenario_splits.json"


@dataclass(frozen=True)
class ScenarioSelection:
    scenario_id: str
    split: str
    base_ids: dict[str, str]
    local_indices: dict[str, int]


class ScenarioCatalog:
    def __init__(self, manifest_path: str | Path = DEFAULT_MANIFEST):
        self.path = Path(manifest_path).resolve()
        contents = self.path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(contents).hexdigest()
        self.manifest = json.loads(contents)
        self._validate_manifest()
        self._verified_paths: dict[tuple[str, str], Path] = {}

    @staticmethod
    def validate_split(split: str) -> str:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, received {split!r}.")
        return split

    def _validate_manifest(self) -> None:
        data = self.manifest
        if data.get("schema_version") != 1 or data.get("benchmark_version") != "benchmark_v1":
            raise ValueError("Unsupported benchmark scenario manifest.")
        if set(data["splits"]) != set(SPLIT_NAMES):
            raise ValueError("Manifest must contain exactly the three official splits.")
        if set(data["sources"]) != set(SOURCE_NAMES):
            raise ValueError("Manifest must contain exactly four base-profile sources.")
        hash_splits: dict[str, set[str]] = {}
        date_splits: dict[str, set[str]] = {}
        self._records: dict[str, dict] = {}
        for source in SOURCE_NAMES:
            records = data["sources"][source]["base_weeks"]
            source_records = {r["base_id"]: r for r in records}
            if len(source_records) != len(records):
                raise ValueError(f"Duplicate base_id in {source}.")
            members = []
            for split in SPLIT_NAMES:
                entry = data["splits"][split]["sources"][source]
                ids = entry["base_ids"]
                if len(ids) != entry["count"] or not ids or len(ids) != len(set(ids)):
                    raise ValueError(f"Invalid count or duplicate member in {split}/{source}.")
                for base_id in ids:
                    if base_id not in source_records:
                        raise ValueError(f"Unknown base profile {base_id}.")
                    record = source_records[base_id]
                    if record["split"] != split:
                        raise ValueError(f"Profile {base_id} belongs to a different split.")
                    hash_splits.setdefault(record["profile_sha256"], set()).add(split)
                    if record["start"] is not None:
                        date_splits.setdefault(record["start"], set()).add(split)
                members.extend(ids)
            if len(members) != len(set(members)) or set(members) != set(source_records):
                raise ValueError(f"Split overlap or incomplete partition in {source}.")
            self._records.update(source_records)
        if any(len(v) != 1 for v in hash_splits.values()):
            raise ValueError("Identical base profiles cross split boundaries.")
        if any(len(v) != 1 for v in date_splits.values()):
            raise ValueError("A dated calendar week crosses split boundaries.")
        for split in SPLIT_NAMES:
            for scenario in data["splits"][split]["reference_scenarios"]:
                self._selection(split, scenario["base_ids"])

    def pool_path(self, split: str, source: str) -> Path:
        self.validate_split(split)
        if source not in SOURCE_NAMES:
            raise ValueError(f"Unknown source {source!r}.")
        key = (split, source)
        if key in self._verified_paths:
            return self._verified_paths[key]
        entry = self.manifest["splits"][split]["sources"][source]
        path = (self.path.parent / entry["path"]).resolve()
        if not path.is_relative_to(self.path.parent):
            raise ValueError("Derived pool path escapes the manifest directory.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Derived pool hash mismatch: {path}. Rebuild or restore the frozen data.")
        with np.load(path, allow_pickle=False) as pool:
            weeks = pool["weeks"]
            ids = pool["base_ids"].tolist()
            if ids != entry["base_ids"] or weeks.shape != (entry["count"], 168):
                raise ValueError(f"Derived pool identity/shape mismatch: {path}.")
            if not np.all(np.isfinite(weeks)) or np.any(weeks < 0):
                raise ValueError(f"Invalid base powers in {path}.")
            for base_id, week in zip(ids, weeks):
                actual = hashlib.sha256(np.asarray(week, dtype="<f8").tobytes()).hexdigest()
                if actual != self._records[base_id]["profile_sha256"]:
                    raise ValueError(f"Base profile hash mismatch for {base_id}.")
        self._verified_paths[key] = path
        return path

    def _selection(self, split: str, bases: Mapping[str, str]) -> ScenarioSelection:
        self.validate_split(split)
        if set(bases) != set(SOURCE_NAMES):
            raise ValueError("A scenario must identify office, wh1, wh2 and solar.")
        indices = {}
        for source in SOURCE_NAMES:
            allowed = self.manifest["splits"][split]["sources"][source]["base_ids"]
            if bases[source] not in allowed:
                raise ValueError(f"{bases[source]!r} is not in split {split!r} for {source}.")
            indices[source] = allowed.index(bases[source])
        canonical = (f"benchmark_v1:{self.manifest_sha256[:12]}:{split}:"
                     + "|".join(bases[source] for source in SOURCE_NAMES))
        return ScenarioSelection(canonical, split, dict(bases), indices)

    def resolve(self, split: str, scenario_id: str) -> ScenarioSelection:
        self.validate_split(split)
        if not isinstance(scenario_id, str):
            raise TypeError("scenario_id must be a string.")
        for ref in self.manifest["splits"][split]["reference_scenarios"]:
            if scenario_id == ref["label"]:
                return self._selection(split, ref["base_ids"])
        prefix = f"benchmark_v1:{self.manifest_sha256[:12]}:{split}:"
        if not scenario_id.startswith(prefix):
            raise ValueError("scenario_id belongs to another split/manifest or is unknown.")
        parts = scenario_id[len(prefix):].split("|")
        if len(parts) != len(SOURCE_NAMES):
            raise ValueError("scenario_id must encode exactly four base profiles.")
        return self._selection(split, dict(zip(SOURCE_NAMES, parts)))

    def sample(self, split: str, rng: np.random.Generator) -> ScenarioSelection:
        self.validate_split(split)
        refs = self.manifest["splits"][split]["reference_scenarios"]
        if refs:
            return self._selection(split, refs[int(rng.integers(len(refs)))]["base_ids"])
        bases = {}
        for source in SOURCE_NAMES:
            allowed = self.manifest["splits"][split]["sources"][source]["base_ids"]
            bases[source] = allowed[int(rng.integers(len(allowed)))]
        return self._selection(split, bases)

    def reference_scenario_ids(self, split: str) -> tuple[str, ...]:
        self.validate_split(split)
        return tuple(self._selection(split, r["base_ids"]).scenario_id
                     for r in self.manifest["splits"][split]["reference_scenarios"])

    def metadata(self, selection: ScenarioSelection) -> dict:
        return {
            "benchmark_version": "benchmark_v1",
            "split": selection.split,
            "scenario_id": selection.scenario_id,
            "base_week_ids": dict(selection.base_ids),
            "base_week_metadata": {
                name: deepcopy(self._records[base_id])
                for name, base_id in selection.base_ids.items()
            },
            "scenario_manifest_sha256": self.manifest_sha256,
            "forecast_model": "synthetic noisy forecasts",
        }

