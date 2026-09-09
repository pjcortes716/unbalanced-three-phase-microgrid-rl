"""Build benchmark_v1 derivatives without overwriting source data.

Run from any directory with the project's Python environment. No learning,
evaluation of policies, or economic-parameter selection takes place here.
"""
from __future__ import annotations

# Permit both direct CLI execution and package imports from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = CODE_DIR / "data" / "raw"  # Optional external preparation input; not distributed.
OUTPUT_DIR = CODE_DIR / "data"
MANIFEST_PATH = CODE_DIR / "data" / "scenario_splits.json"
SPLITS = ("train", "optuna_validation", "final_test")
COLUMNS = {
    "office": "Panther_office_Karla",
    "wh1": "Hog_warehouse_Louise",
    "wh2": "Rat_warehouse_Shari",
}
HOLDOUT_WEEKS = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def write_deterministic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Fixed ZIP timestamps and order make repeated builds byte reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[name], allow_pickle=False)
            entry = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue())


def extract_dated_weeks(frame: pd.DataFrame, source: str):
    """Keep historical office coverage; align every dated source to Monday."""
    column = COLUMNS[source]
    records, values, discarded = [], [], []
    starts = frame["timestamp"].dt.to_period("W-SUN").dt.start_time
    for monday, group in frame.groupby(starts, sort=True):
        group = group.sort_values("timestamp")
        expected = pd.date_range(monday, periods=168, freq="h")
        week = group[column].to_numpy(dtype=float)
        reasons = []
        if len(group) != 168 or not np.array_equal(group["timestamp"].to_numpy(), expected.to_numpy()):
            reasons.append("not_168_unique_consecutive_hours_monday_to_sunday")
        if not np.all(np.isfinite(week)):
            reasons.append("nonfinite_values")
        if np.any(week < 0):
            reasons.append("negative_values")
        if source == "office":
            # Preserve the audited office extractor's weeks[20:] coverage.
            if monday < pd.Timestamp("2016-05-23"):
                reasons.append("outside_legacy_office_coverage_start_2016_05_23")
        else:
            if float(np.mean(week == 0)) > 0.15:
                reasons.append("zero_fraction_gt_0.15")
            if float(np.sum(week)) < 20:
                reasons.append("weekly_sum_lt_20")
        iso = monday.isocalendar()
        record = {
            "base_id": f"{source}-{iso.year}-W{iso.week:02d}",
            "start": monday.isoformat(),
            "end": (monday + pd.Timedelta(hours=167)).isoformat(),
            "calendar": "Monday 00:00 through Sunday 23:00",
            "timezone": None,
            "timestamp_policy": "original naive timestamps; no timezone invented",
            "source_column": column,
            "row_count": len(group),
        }
        if reasons:
            discarded.append({**record, "reasons": reasons})
            continue
        record["profile_sha256"] = profile_hash(week)
        record["aligned_pool_index"] = len(records)
        records.append(record)
        values.append(week)
    return records, np.asarray(values, dtype=float), discarded


def build(electricity: Path, solar_path: Path, output_dir: Path) -> dict:
    """Rebuild the eight distributable pools from explicitly supplied original sources.

    The frozen holdout metadata is retained; final_test arrays are never written.
    Existing output directories are refused to protect the frozen snapshot.
    """
    electricity, solar_path = Path(electricity).resolve(), Path(solar_path).resolve()
    frozen = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if sha256_file(electricity) != frozen["sources"]["office"]["source_sha256"]:
        raise ValueError("Electricity source differs from the approved SHA-256.")
    if sha256_file(solar_path) != frozen["sources"]["solar"]["source_sha256"]:
        raise ValueError("Solar source differs from the approved SHA-256.")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(electricity, usecols=["timestamp", *COLUMNS.values()])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps in the original electricity CSV.")
    frame = frame.sort_values("timestamp")
    source_csv_hash = sha256_file(electricity)
    sources, arrays = {}, {}
    for name in COLUMNS:
        records, weeks, discarded = extract_dated_weeks(frame, name)
        arrays[name] = weeks
        sources[name] = {
            "source_file": frozen["sources"][name]["source_file"],
            "source_sha256": source_csv_hash,
            "source_column": COLUMNS[name],
            "dated": True,
            "filter": {
                "complete_hours": 168, "finite": True, "nonnegative": True,
                "max_zero_fraction": None if name == "office" else 0.15,
                "min_weekly_sum": None if name == "office" else 20.0,
                "minimum_start": "2016-05-23T00:00:00" if name == "office" else None,
            },
            "base_weeks": records, "discarded_weeks": discarded,
        }
    with np.load(solar_path, allow_pickle=False) as data:
        solar = np.asarray(data["weeks"], dtype=float)
    if solar.ndim != 2 or solar.shape[1] != 168:
        raise ValueError("Solar generic profiles must contain 168 values.")
    solar_records, solar_values, solar_discarded = [], [], []
    for index, week in enumerate(solar):
        record = {
            "base_id": f"solar-generic-{index:04d}", "original_index": index,
            "start": None, "end": None, "calendar": "generic weekly profile, dates unknown",
            "timezone": None, "profile_sha256": profile_hash(week),
        }
        if not np.all(np.isfinite(week)) or np.any(week < 0):
            solar_discarded.append({**record, "reasons": ["nonfinite_or_negative_values"]})
        else:
            solar_records.append(record)
            solar_values.append(week)
    arrays["solar"] = np.asarray(solar_values, dtype=float)
    sources["solar"] = {
        "source_file": frozen["sources"]["solar"]["source_file"],
        "source_sha256": sha256_file(solar_path), "dated": False,
        "filter": {"complete_hours": 168, "finite": True, "nonnegative": True},
        "provenance_limit": "No verified extraction or dates; preserve generic profiles unchanged.",
        "base_weeks": solar_records, "discarded_weeks": solar_discarded,
    }
    # Identical base arrays must never silently be put into separate splits.
    for name, source in sources.items():
        hashes = [r["profile_sha256"] for r in source["base_weeks"]]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"{name}: duplicate base profiles require grouping before splitting.")
    common_dates = sorted(set.intersection(*[
        {r["start"] for r in sources[name]["base_weeks"]} for name in COLUMNS
    ]))
    if len(common_dates) < 2 * HOLDOUT_WEEKS or len(solar_records) < 2 * HOLDOUT_WEEKS + 1:
        raise ValueError("Insufficient quality weeks for the requested 6+6 holdouts.")
    val_dates = common_dates[-12:-6]
    test_dates = common_dates[-6:]
    solar_val = {r["base_id"] for r in solar_records[-12:-6]}
    solar_test = {r["base_id"] for r in solar_records[-6:]}
    split_data = {s: {"sources": {}, "reference_scenarios": []} for s in SPLITS}
    for name, source in sources.items():
        members = {s: [] for s in SPLITS}
        for index, record in enumerate(source["base_weeks"]):
            if name == "solar":
                target = ("optuna_validation" if record["base_id"] in solar_val
                          else "final_test" if record["base_id"] in solar_test else "train")
            else:
                target = ("optuna_validation" if record["start"] in val_dates
                          else "final_test" if record["start"] in test_dates else "train")
            record["split"] = target
            members[target].append(index)
        for split, indices in members.items():
            chosen = [source["base_weeks"][i] for i in indices]
            if split == "final_test":
                split_data[split]["sources"][name] = dict(frozen["splits"][split]["sources"][name])
                continue
            path = destination / split / (name + "_weeks_pool.npz")
            write_deterministic_npz(
                path, weeks=arrays[name][indices],
                base_ids=np.asarray([r["base_id"] for r in chosen], dtype="U"),
            )
            split_data[split]["sources"][name] = {
                "path": str(path.relative_to(destination).as_posix()),
                "sha256": sha256_file(path),
                "base_ids": [r["base_id"] for r in chosen],
                "count": len(chosen),
            }
    for split, dates in [("optuna_validation", val_dates), ("final_test", test_dates)]:
        for i, date in enumerate(dates):
            bases = {
                name: next(r["base_id"] for r in sources[name]["base_weeks"] if r["start"] == date)
                for name in COLUMNS
            }
            bases["solar"] = split_data[split]["sources"]["solar"]["base_ids"][i]
            split_data[split]["reference_scenarios"].append({
                "label": f"{split}_{i+1:02d}", "base_ids": bases,
            })
    manifest = {
        "schema_version": 1, "benchmark_version": "benchmark_v1",
        "horizon_hours": 168,
        "split_policy": {
            "base_profile_isolation": True,
            "holdout_weeks_per_source": HOLDOUT_WEEKS,
            "dated_selection": "last 12 common valid Mondays; earlier 6 validation, later 6 final test",
            "solar_selection": "last 12 valid original row indices; generic, not chronological",
            "remaining": "ALL other valid base weeks go to train, separately by source",
            "strict_chronological_split": False,
            "chronology_note": "Some non-common training weeks can be later than holdout dates.",
            "common_dated_weeks": len(common_dates),
            "optuna_validation_mondays": val_dates, "final_test_mondays": test_dates,
            "final_test_policy": "Never use for Optuna, coefficient selection, pruning, or checkpoint selection.",
            "training_scenarios": "independent draws from each source's train-only base weeks",
        },
        "forecast_policy": "synthetic noisy forecasts; extension repeats the SAME chosen base realization",
        "extreme_events_for_initial_studies": None,
        "sources": sources, "splits": split_data,
    }
    (destination / "scenario_splits.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify included pools without raw inputs or writes.")
    parser.add_argument("--electricity-csv", type=Path)
    parser.add_argument("--solar-npz", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        if any((args.electricity_csv, args.solar_npz, args.output_dir)):
            parser.error("--verify-only cannot be combined with rebuild arguments")
        from env.scenario_catalog import ScenarioCatalog, SOURCE_NAMES
        catalog = ScenarioCatalog()
        for split in ("train", "optuna_validation"):
            for source in SOURCE_NAMES:
                catalog.pool_path(split, source)
        if (OUTPUT_DIR / "final_test").exists():
            raise RuntimeError("The distributable snapshot must exclude final_test.")
        print("Verified 8 NPZ files and all included profile hashes; final_test absent.")
    else:
        if not all((args.electricity_csv, args.solar_npz, args.output_dir)):
            parser.error("Use --verify-only, or supply --electricity-csv, --solar-npz and a NEW --output-dir.")
        result = build(args.electricity_csv, args.solar_npz, args.output_dir)
        print(json.dumps({s: {k: v["count"] for k, v in d["sources"].items()}
                          for s, d in result["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
