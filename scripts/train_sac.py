"""CLI for reusable baseline SAC smoke training; never launches Optuna."""

# Permit both direct CLI execution and package imports from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import replace
import json
from pathlib import Path
from scripts.sac_experiment import CODE_DIR, SACExperimentConfig, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CODE_DIR / "experiments" / "sac_smoke_seed42")
    parser.add_argument("--config", type=Path, help="Explicit config JSON, otherwise approved smoke defaults.")
    parser.add_argument("--smoke-reduced", action="store_true", help="Technical-only 2000/500 steps and first two validation scenarios.")
    args = parser.parse_args()
    config = SACExperimentConfig()
    if args.config:
        values = json.loads(args.config.read_text(encoding="utf-8"))
        config = SACExperimentConfig.from_dict(values.get("experiment", values))
    if args.smoke_reduced:
        config = replace(config, total_timesteps=2000, learning_starts=500,
                         validation_scenario_count=2, allow_partial_last_episode=True,
                         purpose="technical_smoke_reduced")
    run_experiment(config, args.output)


if __name__ == "__main__":
    main()

