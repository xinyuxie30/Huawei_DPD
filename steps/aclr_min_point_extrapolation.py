"""Exact lowest-point ACLR line search in model-weight space."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER_PATH = ROOT / "steps" / "aclr_min_point_optimization.py"
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_min_point_extrapolation"


def load_optimizer_module():
    spec = importlib.util.spec_from_file_location("aclr_min_point_optimization", OPTIMIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(path):
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["2284", "998", "786", "796"])
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=5.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--tag", default="scan")
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()

    opt = load_optimizer_module()
    evaluator = opt.load_evaluator()
    anchor = evaluator.load_state_dict(resolve(args.anchor))
    candidate = evaluator.load_state_dict(resolve(args.candidate))
    features, targets, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split_index = int(len(features) * args.split)
    x_val, y_val = features[split_index:], targets[split_index:]
    target_db = opt.NMSE_TARGETS[args.model]

    output_dir = OUT_ROOT / args.tag / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    best_state = None
    best_metrics = None
    best_alpha = None
    for alpha in np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step):
        state = opt.interpolate_states(anchor, candidate, float(alpha))
        model = evaluator.AutoDiagramDPDNet(state).to(evaluator.DEVICE).eval()
        metrics = opt.evaluate_minimum(evaluator, model, x_val, y_val)
        rows.append({"alpha": float(alpha), **metrics})
        if best_metrics is None or opt.candidate_key(metrics, target_db) < opt.candidate_key(
            best_metrics, target_db
        ):
            best_state = state
            best_metrics = metrics
            best_alpha = float(alpha)
            print(
                f"alpha={best_alpha:.2f} NMSE={metrics['nmse_db']:.3f} "
                f"min-ACLR={metrics['aclr_lower_db']:.3f}/"
                f"{metrics['aclr_upper_db']:.3f}/{metrics['aclr_worst_db']:.3f}"
            )

    with (output_dir / "scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(best_state, output_dir / "best.pth")
    summary = {
        "model": args.model,
        "alpha": best_alpha,
        "nmse_target_db": target_db,
        **best_metrics,
        "checkpoint": str(output_dir / "best.pth"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
