import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_SCRIPT = SCRIPT_DIR / "step40.1.6_memory_fir1000.py"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
DEFAULT_INIT = CHECKPOINT_DIR / "EA-TCN_memory_fir1000_best.pth"

SEED = 2026


def load_base_module():
    spec = importlib.util.spec_from_file_location("memory_fir1000", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()
aidpd = base.aidpd
DEVICE = base.DEVICE


BASE_CONFIG = {
    "hidden": 10,
    "input_mode": "rich9",
    "dilations": [1, 2, 4, 8],
    "attention_channels": 3,
    "poly_orders": [1, 3, 5],
    "output_fir_taps": 47,
}


VARIANTS = {
    "baseline": {
        "description": "Known 1000-MAC EA-TCN + 47-tap FIR baseline.",
        "config": dict(BASE_CONFIG, preset="ablation_baseline"),
        "diff_loss_weight": 0.0,
    },
    "fir15_h11": {
        "description": "Move MACs from 47-tap FIR into hidden=11 nonlinear TCN capacity.",
        "config": dict(BASE_CONFIG, preset="ablation_fir15_h11", hidden=11, output_fir_taps=15),
        "diff_loss_weight": 0.0,
    },
    "coprime_dilation": {
        "description": "Replace [1,2,4,8] with co-prime-ish [1,2,5,9] at the same MAC count.",
        "config": dict(BASE_CONFIG, preset="ablation_coprime_dilation", dilations=[1, 2, 5, 9]),
        "diff_loss_weight": 0.0,
    },
    "diff015": {
        "description": "Keep baseline model but add first-difference high-frequency loss.",
        "config": dict(BASE_CONFIG, preset="ablation_diff015"),
        "diff_loss_weight": 0.15,
    },
    "thermal16": {
        "description": "Replace amp7 attention input with causal MA16(|x|^2) thermal proxy.",
        "config": dict(
            BASE_CONFIG,
            preset="ablation_thermal16",
            attention_mode="thermal16",
            thermal_window=16,
        ),
        "diff_loss_weight": 0.0,
    },
    "thermal4_fir39": {
        "description": "Append thermal16 as a fourth attention channel, paid for by FIR47->FIR39.",
        "config": dict(
            BASE_CONFIG,
            preset="ablation_thermal4_fir39",
            attention_channels=4,
            attention_mode="amp37_thermal16",
            thermal_window=16,
            output_fir_taps=39,
        ),
        "diff_loss_weight": 0.0,
    },
    "coprime_thermal": {
        "description": "Combine co-prime dilation and thermal attention without changing FIR length.",
        "config": dict(
            BASE_CONFIG,
            preset="ablation_coprime_thermal",
            dilations=[1, 2, 5, 9],
            attention_mode="thermal16",
            thermal_window=16,
        ),
        "diff_loss_weight": 0.0,
    },
    "all4": {
        "description": "Apply all four proposed changes together; highest risk variant.",
        "config": dict(
            BASE_CONFIG,
            preset="ablation_all4",
            hidden=11,
            dilations=[1, 2, 5, 9],
            output_fir_taps=15,
            attention_mode="thermal16",
            thermal_window=16,
        ),
        "diff_loss_weight": 0.15,
    },
    "deep_rank1_h14_fir29": {
        "description": "Six-layer rank-1 pointwise TCN with long memory and 29-tap FIR, 996 MACs.",
        "config": dict(
            BASE_CONFIG,
            preset="deep_rank1_h14_fir29",
            hidden=14,
            dilations=[1, 2, 4, 8, 16, 32],
            pointwise_rank=1,
            output_fir_taps=29,
        ),
        "diff_loss_weight": 0.0,
    },
    "deep_rank1_h15_fir13": {
        "description": "Six-layer rank-1 pointwise TCN with more channels and short 13-tap FIR, 994 MACs.",
        "config": dict(
            BASE_CONFIG,
            preset="deep_rank1_h15_fir13",
            hidden=15,
            dilations=[1, 2, 4, 8, 16, 32],
            pointwise_rank=1,
            output_fir_taps=13,
        ),
        "diff_loss_weight": 0.0,
    },
    "deep_rank1_base6_h16_fir11": {
        "description": "Six-layer rank-1 pointwise TCN using base6 input, hidden=16 and 11-tap FIR, 1000 MACs.",
        "config": dict(
            BASE_CONFIG,
            preset="deep_rank1_base6_h16_fir11",
            hidden=16,
            input_mode="base6",
            dilations=[1, 2, 4, 8, 16, 32],
            pointwise_rank=1,
            output_fir_taps=11,
        ),
        "diff_loss_weight": 0.0,
    },
    "deep_dense_h9_base6_fir19": {
        "description": "Teacher-shaped six-layer dense TCN: base6 input, hidden=9, 19-tap FIR, 997 MACs.",
        "config": dict(
            BASE_CONFIG,
            preset="deep_dense_h9_base6_fir19",
            hidden=9,
            input_mode="base6",
            dilations=[1, 2, 4, 8, 16, 32],
            output_fir_taps=19,
        ),
        "diff_loss_weight": 0.0,
    },
    "deep_dense_h9_rich9_fir13": {
        "description": "Six-layer dense TCN with rich9 input, hidden=9, 13-tap FIR, exactly 1000 MACs.",
        "config": dict(
            BASE_CONFIG,
            preset="deep_dense_h9_rich9_fir13",
            hidden=9,
            dilations=[1, 2, 4, 8, 16, 32],
            output_fir_taps=13,
        ),
        "diff_loss_weight": 0.0,
        "prune_init": True,
    },
    "deep_dense_h8_rich9_fir51": {
        "description": "Six-layer dense TCN with rich9 input, hidden=8, long 51-tap FIR, exactly 1000 MACs.",
        "config": dict(
            BASE_CONFIG,
            preset="deep_dense_h8_rich9_fir51",
            hidden=8,
            dilations=[1, 2, 4, 8, 16, 32],
            output_fir_taps=51,
        ),
        "diff_loss_weight": 0.0,
        "prune_init": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="EA-TCN ablation runner for <=1000 real multipliers."
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="fir15_h11")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT))
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--epochs-base", type=int, default=260)
    parser.add_argument("--epochs-long", type=int, default=160)
    parser.add_argument("--base-lr", type=float, default=8e-4)
    parser.add_argument("--long-lr", type=float, default=2e-6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--long-batch-size", type=int, default=4)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--stride-length", type=int, default=32)
    parser.add_argument("--long-frame-length", type=int, default=4096)
    parser.add_argument("--long-stride-length", type=int, default=256)
    parser.add_argument("--diff-loss-weight", type=float, default=None)
    parser.add_argument("--skip-initial-ls", action="store_true")
    parser.add_argument("--target-fullseq-db", type=float, default=-40.0)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--main-band", type=float, nargs=2, default=(-100.0, 100.0))
    parser.add_argument("--lower-band", type=float, nargs=2, default=(-245.0, -180.0))
    parser.add_argument("--upper-band", type=float, nargs=2, default=(180.0, 245.0))
    parser.add_argument("--aclr-target-dbc", type=float, default=-45.0)
    return parser.parse_args()


def default_checkpoint_for_variant(variant):
    return CHECKPOINT_DIR / f"EA-TCN_{variant}_ablation1000_best.pth"


def load_pruned_same_family(model, checkpoint_path):
    source = aidpd.load_state_dict(checkpoint_path)
    target = model.state_dict()
    src_h = int(source["in_conv.weight"].shape[0])
    dst_h = int(target["in_conv.weight"].shape[0])
    if dst_h >= src_h:
        raise ValueError("Pruned init expects destination hidden smaller than source hidden.")
    if source["in_conv.weight"].shape[1] != target["in_conv.weight"].shape[1]:
        raise ValueError("Pruned init requires the same input feature mode.")

    scores = torch.zeros(src_h, dtype=torch.float64)
    score_keys = ["in_conv.weight", "out_conv.weight", "attn.env_conv.weight"]
    for key in score_keys:
        value = source[key].to(dtype=torch.float64)
        if key == "out_conv.weight":
            scores += torch.linalg.vector_norm(value[:, :, :], dim=(0, 2))
        else:
            scores += torch.linalg.vector_norm(value, dim=tuple(range(1, value.ndim)))
    for key, value in source.items():
        if key.startswith("tcn_blocks.") and key.endswith(".0.weight"):
            scores += torch.linalg.vector_norm(value.to(dtype=torch.float64), dim=(1, 2))
        if key.startswith("tcn_blocks.") and key.endswith(".1.weight"):
            matrix = value[:, :, 0].to(dtype=torch.float64)
            scores += torch.linalg.vector_norm(matrix, dim=0)
            scores += torch.linalg.vector_norm(matrix, dim=1)

    keep = torch.topk(scores, dst_h).indices.sort().values
    keep_list = [int(item) for item in keep.tolist()]
    new_state = {key: value.clone() for key, value in target.items()}

    scalar_keys = ["g_real", "g_imag", "g3_real", "g3_imag", "g5_real", "g5_imag"]
    for key in scalar_keys:
        if key in source and key in new_state:
            new_state[key] = source[key].to(dtype=new_state[key].dtype)

    new_state["in_conv.weight"] = source["in_conv.weight"][keep, :, :].to(dtype=new_state["in_conv.weight"].dtype)
    new_state["attn.env_conv.weight"] = source["attn.env_conv.weight"][keep, :, :].to(
        dtype=new_state["attn.env_conv.weight"].dtype
    )
    new_state["out_conv.weight"] = source["out_conv.weight"][:, keep, :].to(dtype=new_state["out_conv.weight"].dtype)

    src_blocks = sorted(
        {
            int(key.split(".")[1])
            for key in source
            if key.startswith("tcn_blocks.") and key.endswith(".0.weight")
        }
    )
    dst_blocks = sorted(
        {
            int(key.split(".")[1])
            for key in target
            if key.startswith("tcn_blocks.") and key.endswith(".0.weight")
        }
    )
    for block_id in dst_blocks:
        depth_key = f"tcn_blocks.{block_id}.0.weight"
        point_key = f"tcn_blocks.{block_id}.1.weight"
        if block_id in src_blocks:
            new_state[depth_key] = source[depth_key][keep, :, :].to(dtype=new_state[depth_key].dtype)
            point = source[point_key][keep][:, keep, :].to(dtype=new_state[point_key].dtype)
            new_state[point_key] = point
        else:
            new_state[point_key] = torch.zeros_like(new_state[point_key])

    model.load_state_dict(new_state, strict=True)
    return keep_list


def plot_history(histories, out_path, title):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    offset = 0
    for stage_name, history in histories:
        xs = [offset + item[0] for item in history]
        ys = [item[1] for item in history]
        plt.plot(xs, ys, label=stage_name)
        offset = xs[-1]
    plt.axhline(-37.0, color="black", linestyle="--", linewidth=1.1, label="-37 dB")
    plt.axhline(-40.0, color="tab:red", linestyle=":", linewidth=1.1, label="-40 dB")
    plt.xlabel("Epoch")
    plt.ylabel("Full-sequence NMSE (dB)")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    variant = VARIANTS[args.variant]
    config = dict(variant["config"])
    diff_loss_weight = variant["diff_loss_weight"] if args.diff_loss_weight is None else args.diff_loss_weight
    checkpoint_path = base.resolve_path(args.checkpoint or default_checkpoint_for_variant(args.variant))
    init_checkpoint = base.resolve_path(args.init_checkpoint)
    data_path = base.resolve_path(args.data)

    x_train, y_train, x_val, y_val, g_real, g_imag = base.load_data(data_path)
    multipliers = aidpd.real_multiplier_count_from_config(config)
    print(f">>> Variant: {args.variant}")
    print(f">>> Description: {variant['description']}")
    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Data: {data_path}")
    print(f">>> Validation samples: {len(x_val)}")
    print(f">>> Model config: {config}")
    print(f">>> Real multipliers: {multipliers}")
    print(f">>> Diff loss weight: {diff_loss_weight}")
    if multipliers > 1000:
        raise RuntimeError(f"Multiplier budget exceeded: {multipliers}")

    model = aidpd.DiagramDPDNet(g_real, g_imag, config=config).to(DEVICE)
    if args.evaluate_only:
        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
        nmse, nmse_db = base.fullseq_nmse(model, x_val, y_val)
        print(f">>> Full-sequence NMSE: {nmse:.6e} ({nmse_db:.2f} dB)")
        aidpd.evaluate_aclr_visualization(
            model,
            x_val,
            y_val,
            output_prefix=checkpoint_path.stem,
            fs_mhz=args.fs_mhz,
            main_band=tuple(args.main_band),
            lower_band=tuple(args.lower_band),
            upper_band=tuple(args.upper_band),
            target_dbc=args.aclr_target_dbc,
        )
        return

    if not args.scratch and init_checkpoint.exists() and variant.get("prune_init"):
        keep_channels = load_pruned_same_family(model, init_checkpoint)
        print(f">>> Saliency-pruned init from: {init_checkpoint}")
        print(f">>> Kept source channels: {keep_channels}")
    elif not args.scratch and init_checkpoint.exists():
        loaded, partial, skipped = aidpd.load_partial_state(model, init_checkpoint)
        print(f">>> Partial init from: {init_checkpoint}")
        print(f">>> Loaded tensors: exact={len(loaded)}, partial={len(partial)}, skipped={len(skipped)}")
    elif args.scratch:
        print(">>> Scratch mode: no checkpoint initialization.")
    else:
        print(f">>> Init checkpoint not found, training fresh: {init_checkpoint}")

    if not args.skip_initial_ls:
        print(f"\n>>> Initial train-only LS FIR fit ({config['output_fir_taps']} taps).")
        aidpd.fit_output_fir_from_train(model, x_train, y_train)
        init_nmse, init_db = base.fullseq_nmse(model, x_val, y_val)
        print(f">>> After initial LS FIR full-sequence NMSE: {init_nmse:.6e} ({init_db:.2f} dB)")

    histories = []
    h1, _, _ = base.train_stage(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        stage_name="short-frame",
        epochs=args.epochs_base,
        lr=args.base_lr,
        frame_length=args.frame_length,
        stride_length=args.stride_length,
        batch_size=args.batch_size,
        diff_loss_weight=diff_loss_weight,
        checkpoint_path=checkpoint_path,
        target_fullseq_db=args.target_fullseq_db,
    )
    histories.append(("short-frame", h1))

    print(f"\n>>> Fitting {config['output_fir_taps']}-tap complex output FIR by train-only LS.")
    aidpd.fit_output_fir_from_train(model, x_train, y_train)
    torch.save(model.state_dict(), checkpoint_path)
    fir_nmse, fir_db = base.fullseq_nmse(model, x_val, y_val)
    print(f">>> After LS FIR full-sequence NMSE: {fir_nmse:.6e} ({fir_db:.2f} dB)")

    if fir_db > args.target_fullseq_db and args.epochs_long > 0:
        base.set_fir_trainable(model, False)
        h2, _, _ = base.train_stage(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            stage_name="long-frame-freeze-fir",
            epochs=args.epochs_long,
            lr=args.long_lr,
            frame_length=args.long_frame_length,
            stride_length=args.long_stride_length,
            batch_size=args.long_batch_size,
            diff_loss_weight=0.0,
            checkpoint_path=checkpoint_path,
            target_fullseq_db=args.target_fullseq_db,
        )
        histories.append(("long-frame-freeze-fir", h2))

        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
        print("\n>>> Final train-only LS FIR refit.")
        aidpd.fit_output_fir_from_train(model, x_train, y_train)
        torch.save(model.state_dict(), checkpoint_path)

    final_nmse, final_db = base.fullseq_nmse(model, x_val, y_val)
    print(f"\n>>> Final full-sequence NMSE: {final_nmse:.6e} ({final_db:.2f} dB)")
    print(f">>> Saved checkpoint: {checkpoint_path}")

    convergence_path = FIGURE_DIR / f"{checkpoint_path.stem}_convergence.png"
    plot_history(histories, convergence_path, f"EA-TCN ablation {args.variant} ({multipliers} multipliers)")
    print(f">>> Convergence figure: {convergence_path}")

    aidpd.evaluate_aclr_visualization(
        model,
        x_val,
        y_val,
        output_prefix=checkpoint_path.stem,
        fs_mhz=args.fs_mhz,
        main_band=tuple(args.main_band),
        lower_band=tuple(args.lower_band),
        upper_band=tuple(args.upper_band),
        target_dbc=args.aclr_target_dbc,
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
