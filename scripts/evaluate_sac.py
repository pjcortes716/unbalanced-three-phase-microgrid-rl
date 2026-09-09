"""Six-scenario deterministic validation CLI; final_test is forbidden."""

# Permit both direct CLI execution and package imports from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
import torch
from scripts.sac_experiment import (
    CheckedSAC, CsvSink, DataAccessAudit, SACExperimentConfig, assert_model_finite,
    evaluate_validation, validate_tuning_split, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="NEW directory, never overwritten.")
    parser.add_argument("--split", default="optuna_validation")
    args = parser.parse_args()
    validate_tuning_split(args.split)
    if args.split != "optuna_validation":
        parser.error("This evaluator requires optuna_validation.")
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    config = SACExperimentConfig.from_dict(raw.get("experiment", raw))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.torch_threads)
    torch.use_deterministic_algorithms(True)
    sink = CsvSink(args.output / "validation.csv")
    try:
        with DataAccessAudit(args.output / "data_access.jsonl") as audit:
            with audit.phase("validation_external"):
                model = CheckedSAC.load(args.model, device=config.device)
                assert_model_finite(model)
                rows, _ = evaluate_validation(model, config, sink=sink)
            write_json(args.output / "evaluation_summary.json",
                       {"episodes": len(rows), "deterministic": True, "data_access": audit.summary()})
    finally:
        sink.close()


if __name__ == "__main__":
    main()

