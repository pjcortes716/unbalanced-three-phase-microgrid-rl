"""Build a whitelist-only VM bundle without opening final_test timeseries."""
from __future__ import annotations

# Permit both direct CLI execution and package imports from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

CODE_DIR = Path(__file__).resolve().parents[1]
CODE_FILES = ('env/Ambiente_industrial.py', 'env/battery.py', 'env/benchmark_metrics.py', 'env/extreme_events.py', 'env/oficinas.py', 'env/opendss_easy.py', 'env/price_sampler.py', 'env/PV_plant.py', 'env/scenario_catalog.py', 'env/v2g_bus_5.py', 'env/vehiculos_Zoe.py', 'scripts/prepare_benchmark_v1_data.py', 'scripts/run_benchmark_v1_checks.py', 'scripts/sac_experiment.py', 'scripts/train_sac.py', 'scripts/evaluate_sac.py', 'scripts/sac_report.py', 'optuna/vm_campaign_config.py', 'optuna/vm_campaign_worker.py', 'optuna/vm_campaign.py', 'optuna/vm_campaign.json', 'optuna/prepare_vm_bundle.py', 'optuna/requirements_vm.txt', 'optuna/OPTUNA_VM_CAMPAIGN.md', 'tests/test_benchmark_v1.py', 'tests/test_action_resolution.py', 'tests/test_electrical_instrumentation.py', 'tests/test_electrical_snapshot_2e.py', 'tests/test_environment_reset_2c.py', 'tests/test_environment_step.py', 'tests/test_exogenous_realization_layers.py', 'tests/test_load_telemetry.py', 'tests/test_opendss_easy_1c_b.py', 'tests/test_price_sampler.py', 'tests/test_pv_action_decoder_2d.py', 'tests/test_pv_spatial_variation_2d1.py', 'tests/test_reward.py', 'tests/test_temporal_contract_2b.py', 'tests/test_transformer_production.py', 'tests/test_vehiculos_zoe.py', 'tests/test_vminpu_constant_power.py', 'tests/test_episode_render.py', 'BENCHMARK_V1_CHANGELOG.md', 'AUDITORIA_AMBIENTE_TRIFASICO.md', 'data/scenario_splits.json', 'env/__init__.py', 'scripts/__init__.py', 'README.md', 'LICENSE', '.gitignore', 'requirements.txt', 'DATA_PROVENANCE.md')
DATA_FILES = ('data/train/office_weeks_pool.npz', 'data/train/wh1_weeks_pool.npz', 'data/train/wh2_weeks_pool.npz', 'data/train/solar_weeks_pool.npz', 'data/optuna_validation/office_weeks_pool.npz', 'data/optuna_validation/wh1_weeks_pool.npz', 'data/optuna_validation/wh2_weeks_pool.npz', 'data/optuna_validation/solar_weeks_pool.npz')


def build_bundle(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = CODE_FILES + DATA_FILES
    source_manifest = json.loads((CODE_DIR / "data" / "scenario_splits.json").read_text(encoding="utf-8"))
    for split in ("train", "optuna_validation"):
        for source in ("office", "wh1", "wh2", "solar"):
            entry = source_manifest["splits"][split]["sources"][source]
            path = CODE_DIR / "data" / entry["path"]
            if path.resolve() != (CODE_DIR / f"data/{split}/{source}_weeks_pool.npz").resolve():
                raise ValueError("Unexpected pool path outside the deployment whitelist.")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Approved pool hash mismatch: {split}/{source}")
    hashes = {}
    with zipfile.ZipFile(destination, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            if "final_test" in Path(relative).parts:
                raise ValueError("final_test datasets cannot enter the VM bundle.")
            payload = (CODE_DIR / relative).read_bytes()
            hashes[relative] = hashlib.sha256(payload).hexdigest()
            archive.writestr("unbalanced-three-phase-microgrid-rl/" + relative, payload)
        manifest = {"files_sha256": hashes, "timeseries_splits": ["train", "optuna_validation"],
                    "final_test_timeseries_included": False,
                    "metadata_note": "Snapshot manifest uses portable paths; all profile and NPZ hashes are unchanged. Holdout metadata is not timeseries access.",
                    "execution_status": "Prepared locally; VM preflight and optimization not executed during packaging."}
        archive.writestr("unbalanced-three-phase-microgrid-rl/bundle_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Bundle CRC validation failed.")
        if any("/final_test/" in name for name in archive.namelist()):
            raise RuntimeError("Forbidden holdout dataset found in deployment bundle.")
        for relative, expected in hashes.items():
            actual = hashlib.sha256(archive.read("unbalanced-three-phase-microgrid-rl/" + relative)).hexdigest()
            if actual != expected:
                raise RuntimeError("Bundle SHA-256 validation failed.")
    result = {"path": str(destination), "bytes": destination.stat().st_size,
              "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
              "file_count": len(hashes) + 1, "timeseries_files": len(DATA_FILES), "final_test_timeseries_files": 0}
    destination.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    print(json.dumps(build_bundle(parser.parse_args().output), indent=2))

