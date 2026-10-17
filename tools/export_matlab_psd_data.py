"""Export the final model signals for the MATLAB PSD plotting script."""

import argparse
import importlib.util
from pathlib import Path

import scipy.io


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = PROJECT_ROOT / "steps" / "final_evaluate_unified.py"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "eval_outputs_direct"
    / "aclr_min_point_extrapolation"
    / "scan_2284"
    / "2284"
    / "best.pth"
)


def load_evaluator():
    spec = importlib.util.spec_from_file_location("final_evaluate_unified", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "DPD_data.mat")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval_outputs_direct" / "matlab_psd" / "model_psd_data.mat",
    )
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    args = parser.parse_args()

    evaluator = load_evaluator()
    features, targets, dataset_errors, *_ = evaluator.load_iq_arrays(args.data, "rms")
    split_idx = int(min(len(features), len(targets), len(dataset_errors)) * args.split)
    x_val = features[split_idx:]
    y_val = targets[split_idx:]
    dataset_error_val = dataset_errors[split_idx:]

    state_dict = evaluator.load_state_dict(args.checkpoint)
    model = evaluator.AutoDiagramDPDNet(state_dict).to(evaluator.DEVICE).eval()
    pred_iq, target_iq, nmse_linear, nmse_db = evaluator.evaluate_full_sequence(
        model, x_val, y_val
    )

    target_y = evaluator.tensor_to_complex(target_iq)
    pred_y = evaluator.tensor_to_complex(pred_iq)
    model_error = target_y - pred_y
    dataset_error = evaluator.tensor_to_complex(dataset_error_val.T)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        args.output,
        {
            "target_y": target_y.reshape(-1, 1),
            "pred_y": pred_y.reshape(-1, 1),
            "model_error": model_error.reshape(-1, 1),
            "dataset_error": dataset_error.reshape(-1, 1),
            "FsBB": args.fs_mhz * 1e6,
            "nmse_linear": nmse_linear,
            "nmse_db": nmse_db,
            "checkpoint_name": args.checkpoint.name,
        },
    )
    print(f"Saved MATLAB PSD data: {args.output}")
    print(f"Validation samples: {len(target_y)} | NMSE: {nmse_db:.4f} dB")


if __name__ == "__main__":
    main()
