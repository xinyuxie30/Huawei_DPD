"""Transplant a phase-spline wrapper checkpoint into a larger-budget architecture.

Builds a target wrapper (different spline lag set and/or FIR tap count) from the
release base state, then copies from a source wrapper checkpoint:

* every ``base.*`` parameter (base architecture is unchanged);
* the amplitude spline-table row and the Re/Im rows of every lag shared with
  the source topology (new lags start at zero);
* the output FIR, center-placed (zero-padded or center-cropped to the target
  tap count).

Zero-padding the FIR and zero-initializing new lag rows leave the deployed
input-output map identical to the source, so the transplant reproduces the
source metrics exactly while parking the extra MAC head-room at zero.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def table_row_indices(offsets):
    """Row layout: [amp] + (Re, Im) per non-zero offset in list order."""
    rows = {"amp": 0}
    cursor = 1
    for offset in offsets:
        if offset == 0:
            continue
        rows[(offset, "re")] = cursor
        rows[(offset, "im")] = cursor + 1
        cursor += 2
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-offsets", type=int, nargs="+", required=True)
    parser.add_argument("--source-taps-reduction", type=int, required=True)
    parser.add_argument("--target-offsets", type=int, nargs="+", required=True)
    parser.add_argument("--target-taps-reduction", type=int, required=True)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--mac-limit", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()

    evaluator = load_module("final_evaluate_unified", "steps/final_evaluate_unified.py")
    metrics_mod = load_module("aclr_ref_metrics", "steps/aclr_ref_metrics.py")
    ps_mod = load_module("aclr_phase_spline_experiment", "steps/aclr_phase_spline_experiment.py")
    mp_mod = load_module("aclr_min_point_optimization", "steps/aclr_min_point_optimization.py")

    features, targets, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split = int(len(features) * args.split)
    amp_max = 1.05 * float(np.max(np.hypot(features[:, 0], features[:, 1])))

    source_path = Path(args.source_checkpoint)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source_state = evaluator.load_state_dict(source_path)
    release_state = evaluator.load_state_dict(ROOT / metrics_mod.MODEL_SPECS[args.model][0])

    target = ps_mod.PhaseSplineDPD(
        evaluator, release_state, args.target_taps_reduction, args.target_offsets, args.grid, amp_max
    )
    with torch.no_grad():
        for key, value in source_state.items():
            if key.startswith("base."):
                target.state_dict()[key].copy_(value)
        src_rows = table_row_indices(args.source_offsets)
        dst_rows = table_row_indices(args.target_offsets)
        src_tables = source_state["corr.tables"]
        target.corr.tables.zero_()
        for row_key, dst_index in dst_rows.items():
            if row_key in src_rows:
                target.corr.tables[dst_index].copy_(src_tables[src_rows[row_key]])
        src_fir_r = source_state["fir_real"]
        src_fir_i = source_state["fir_imag"]
        t_taps = len(target.fir_real)
        s_taps = len(src_fir_r)
        target.fir_real.zero_()
        target.fir_imag.zero_()
        if t_taps >= s_taps:
            lo = (t_taps - s_taps) // 2
            target.fir_real[lo : lo + s_taps].copy_(src_fir_r)
            target.fir_imag[lo : lo + s_taps].copy_(src_fir_i)
        else:
            lo = (s_taps - t_taps) // 2
            target.fir_real.copy_(src_fir_r[lo : lo + t_taps])
            target.fir_imag.copy_(src_fir_i[lo : lo + t_taps])

    macs = target.mac_report(evaluator, release_state, strict=True)
    print(
        f"[{args.model}] transplant MAC: base {macs['base_no_fir']} + FIR {macs['fir']} + "
        f"corr {macs['corr']} = {macs['total']} (limit {args.mac_limit})"
    )
    if macs["total"] > args.mac_limit:
        raise SystemExit("MAC limit exceeded")

    target = target.to(evaluator.DEVICE).eval()
    check = mp_mod.evaluate_minimum(evaluator, target, features[split:], targets[split:])
    print(
        f"[{args.model}] transplant check: NMSE {check['nmse_db']:.3f} | "
        f"point {check['aclr_lower_db']:.3f}/{check['aclr_upper_db']:.3f}/{check['aclr_worst_db']:.3f}"
    )
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in target.state_dict().items()}, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
