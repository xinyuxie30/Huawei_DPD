"""Metric-guided weight extrapolation and two-direction task-vector search.

The model architecture is reconstructed unchanged from compatible checkpoints.
For a checkpoint chain B, C1, C2, candidates are

    B + a * (C1 - B) + b * (C2 - C1).

Each candidate is accepted/ranked with the exact validation Welch metrics and an
NMSE hard constraint.  This deliberately searches beyond ordinary interpolation
(``a`` or ``b`` greater than one).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
METRIC_SCRIPT = ROOT / "steps" / "aclr_three_metric_experiment.py"
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_weight_space_search"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metric_exp = load_module("aclr_three_metric_experiment", METRIC_SCRIPT)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def grid_values(bounds):
    low, high, step = map(float, bounds)
    return np.arange(low, high + step / 2.0, step)


def chain_state(states, coefficients):
    result = {}
    for key, base_value in states[0].items():
        if not torch.is_floating_point(base_value):
            result[key] = states[-1][key].clone()
            continue
        value = base_value.clone()
        for index, coefficient in enumerate(coefficients):
            value.add_(states[index + 1][key] - states[index][key], alpha=float(coefficient))
        result[key] = value
    return result


def reference_metrics(evaluator, targets, dataset_errors):
    target = evaluator.tensor_to_complex(targets.T)
    error = evaluator.tensor_to_complex(dataset_errors.T)
    freq, target_psd = evaluator.psd_db(target, 491.52e6, 4096)
    _, error_psd = evaluator.psd_db(error, 491.52e6, 4096)
    full = evaluator.aclr_from_psd(
        freq, error_psd, target_psd, (20.0, 80.0), (-180.0, -120.0), (120.0, 180.0)
    )
    point = evaluator.aclr_from_psd_minimum(
        freq, error_psd, target_psd, (20.0, 80.0), (-180.0, -120.0), (120.0, 180.0)
    )
    narrow = evaluator.aclr_from_psd(
        freq,
        error_psd,
        target_psd,
        evaluator.NARROW_MAIN_BAND_MHZ,
        evaluator.NARROW_LOWER_BAND_MHZ,
        evaluator.NARROW_UPPER_BAND_MHZ,
    )
    return {"full": full[2], "point": point[2], "narrow": narrow[2]}


def objective_value(metrics, objective, reference):
    values = [metrics["full_worst"], metrics["point_worst"], metrics["narrow_worst"]]
    if objective == "reference":
        return max(
            metrics["full_worst"] - reference["full"],
            metrics["point_worst"] - reference["point"],
            metrics["narrow_worst"] - reference["narrow"],
        )
    if objective == "target45":
        return max(value + 45.0 for value in values)
    return max(metrics["full_worst"], metrics["narrow_worst"])


def candidate_key(metrics, nmse_target, objective, reference):
    violation = max(0.0, metrics["nmse_db"] - nmse_target)
    return (
        violation > 0.0,
        violation,
        objective_value(metrics, objective, reference),
        metrics["full_worst"],
        metrics["narrow_worst"],
    )


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["2284", "998", "786", "796"])
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--coeff1", nargs=3, type=float, default=(0.0, 3.0, 0.1), metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--coeff2", nargs=3, type=float, default=(-0.5, 3.0, 0.1), metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--objective", choices=["reference", "target45", "stable"], default="reference")
    parser.add_argument("--tag", default="search")
    parser.add_argument("--split", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.checkpoints) not in {2, 3}:
        raise ValueError("Use two checkpoints for 1-D extrapolation or three for a 2-D chain search")
    evaluator = metric_exp.load_evaluator()
    features, targets, errors, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split_index = int(len(features) * args.split)
    x_val, y_val, error_val = features[split_index:], targets[split_index:], errors[split_index:]
    reference = reference_metrics(evaluator, y_val, error_val)
    spec = metric_exp.MODEL_SPECS[args.model]
    paths = [resolve(path) for path in args.checkpoints]
    states = [evaluator.load_state_dict(path) for path in paths]
    if any(state.keys() != states[0].keys() for state in states[1:]):
        raise ValueError("All checkpoints must have exactly the same architecture/state keys")

    coefficient_grids = [grid_values(args.coeff1)]
    if len(states) == 3:
        coefficient_grids.append(grid_values(args.coeff2))
    output_dir = OUT_ROOT / args.tag / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {**vars(args), "checkpoints": [str(path) for path in paths], "reference": reference},
            handle,
            indent=2,
            default=str,
        )

    best_state = None
    best_metrics = None
    best_coefficients = None
    rows = []
    for coefficients in itertools.product(*coefficient_grids):
        state = chain_state(states, coefficients)
        model = evaluator.AutoDiagramDPDNet(state).to(evaluator.DEVICE).eval()
        metrics = metric_exp.evaluate_metrics(evaluator, model, x_val, y_val)
        del model
        score = objective_value(metrics, args.objective, reference)
        row = {
            "model": args.model,
            "coefficient1": float(coefficients[0]),
            "coefficient2": float(coefficients[1]) if len(coefficients) > 1 else "",
            "objective": args.objective,
            "objective_value": score,
            **metrics,
        }
        rows.append(row)
        if best_metrics is None or candidate_key(
            metrics, spec.nmse_target_db, args.objective, reference
        ) < candidate_key(best_metrics, spec.nmse_target_db, args.objective, reference):
            best_state = state
            best_metrics = metrics
            best_coefficients = coefficients
            print(
                f"best coeff={tuple(round(float(c), 3) for c in coefficients)} "
                f"NMSE={metrics['nmse_db']:.3f} ACLR={metrics['full_worst']:.3f}/"
                f"{metrics['point_worst']:.3f}/{metrics['narrow_worst']:.3f} score={score:.3f}"
            )

    write_rows(output_dir / "search.csv", rows)
    torch.save(best_state, output_dir / "best.pth")
    summary = {
        "model": args.model,
        "coefficients": [float(value) for value in best_coefficients],
        "objective": args.objective,
        "objective_value": objective_value(best_metrics, args.objective, reference),
        "nmse_target_db": spec.nmse_target_db,
        "reference": reference,
        "metrics": best_metrics,
        "checkpoint": str(output_dir / "best.pth"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
