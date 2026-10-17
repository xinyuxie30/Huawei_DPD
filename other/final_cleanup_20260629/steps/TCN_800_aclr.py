import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AI_DPD_PATH = SCRIPT_DIR / "AI-DPD.py"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
DEFAULT_INIT = (
    PROJECT_ROOT
    / "best_raw_model_A.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2026


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd", AI_DPD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd_module()


VARIANTS = {
    "h10_base6_l3_fir41": {
        "description": "Keep legacy H10 channels, remove one TCN block, and spend the budget on a 41-tap FIR.",
        "config": {
            "preset": "h10_base6_l3_fir41_796",
            "hidden": 10,
            "input_mode": "base6",
            "dilations": [1, 2, 4],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 41,
        },
        "init": "partial",
    },
    "h10_base6_l3_poly13_fir37": {
        "description": "H10/L3 with explicit high-order static nonlinearity and a 37-tap FIR.",
        "config": {
            "preset": "h10_base6_l3_poly13_fir37_796",
            "hidden": 10,
            "input_mode": "base6",
            "dilations": [1, 2, 4],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5, 7, 9, 11, 13],
            "output_fir_taps": 37,
        },
        "init": "partial",
    },
    "h10_rich9_l3_fir33": {
        "description": "H10/L3 plus rich9 local memory features and a 33-tap FIR.",
        "config": {
            "preset": "h10_rich9_l3_fir33_794",
            "hidden": 10,
            "input_mode": "rich9",
            "dilations": [1, 2, 4],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 33,
        },
        "init": "partial",
    },
    "h10_rich9_l3_poly9_fir31": {
        "description": "H10/L3 rich9 with 7/9 static terms and a 31-tap FIR.",
        "config": {
            "preset": "h10_rich9_l3_poly9_fir31_794",
            "hidden": 10,
            "input_mode": "rich9",
            "dilations": [1, 2, 4],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5, 7, 9],
            "output_fir_taps": 31,
        },
        "init": "partial",
    },
    "h10_base6_l2_fir79": {
        "description": "Very long FIR budget: H10/L2 trunk with a 79-tap output FIR.",
        "config": {
            "preset": "h10_base6_l2_fir79_798",
            "hidden": 10,
            "input_mode": "base6",
            "dilations": [1, 2],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 79,
        },
        "init": "partial",
    },
    "h11_base6_l3_fir17": {
        "description": "More nonlinear channel capacity at H11/L3, with a shorter 17-tap FIR.",
        "config": {
            "preset": "h11_base6_l3_fir17_795",
            "hidden": 11,
            "input_mode": "base6",
            "dilations": [1, 2, 4],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 17,
        },
        "init": "partial",
    },
    "h9_base6_l4_fir31": {
        "description": "Prune H10/FIR53 994-MAC baseline to H9 and keep a 31-tap output FIR.",
        "config": {
            "preset": "h9_base6_l4_fir31_793",
            "hidden": 9,
            "input_mode": "base6",
            "dilations": [1, 2, 4, 8],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 31,
        },
        "init": "prune994",
    },
    "h9_base6_l4_poly13_fir27": {
        "description": "Trade four FIR taps for explicit 7/9/11/13 order static nonlinear terms.",
        "config": {
            "preset": "h9_base6_l4_poly13_fir27_793",
            "hidden": 9,
            "input_mode": "base6",
            "dilations": [1, 2, 4, 8],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5, 7, 9, 11, 13],
            "output_fir_taps": 27,
        },
        "init": "prune994",
    },
    "h8_base6_l4_fir59": {
        "description": "Aggressive spectrum model: H8 nonlinear trunk with 59-tap output FIR.",
        "config": {
            "preset": "h8_base6_l4_fir59_800",
            "hidden": 8,
            "input_mode": "base6",
            "dilations": [1, 2, 4, 8],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 59,
        },
        "init": "prune994",
    },
    "h8_rich9_l4_fir53": {
        "description": "Keep rich9 memory features while shrinking hidden to H8; long 53-tap FIR.",
        "config": {
            "preset": "h8_rich9_l4_fir53_800",
            "hidden": 8,
            "input_mode": "rich9",
            "dilations": [1, 2, 4, 8],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 53,
        },
        "init": "prune994",
    },
    "h8_base6_l5_fir33": {
        "description": "Add one TCN memory layer at H8 and reserve 33 FIR taps.",
        "config": {
            "preset": "h8_base6_l5_fir33_800",
            "hidden": 8,
            "input_mode": "base6",
            "dilations": [1, 2, 4, 8, 16],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 33,
        },
        "init": "prune994",
    },
    "h7_rich9_l6_fir35": {
        "description": "Six-layer long-memory rich9 trunk with a moderate 35-tap FIR.",
        "config": {
            "preset": "h7_rich9_l6_fir35_796",
            "hidden": 7,
            "input_mode": "rich9",
            "dilations": [1, 2, 4, 8, 16, 32],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 35,
        },
        "init": "prune994",
    },
    "h10_rank1_rich9_l6_fir41": {
        "description": "Long-memory H10 trunk with rank-1 pointwise mixing and 41-tap FIR.",
        "config": {
            "preset": "h10_rank1_rich9_l6_fir41_796",
            "hidden": 10,
            "input_mode": "rich9",
            "dilations": [1, 2, 4, 8, 16, 32],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "pointwise_rank": 1,
            "output_fir_taps": 41,
        },
        "init": "partial",
    },
    "h11_rank1_rich9_l6_fir25": {
        "description": "More rank-1 channels with six-layer memory and shorter 25-tap FIR.",
        "config": {
            "preset": "h11_rank1_rich9_l6_fir25_794",
            "hidden": 11,
            "input_mode": "rich9",
            "dilations": [1, 2, 4, 8, 16, 32],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "pointwise_rank": 1,
            "output_fir_taps": 25,
        },
        "init": "partial",
    },
}


def resolve_path(path_like):
    if path_like is None:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return PROJECT_ROOT / path


def load_data(data_path):
    _, _, x_c, y_c = aidpd.load_normalized_data(data_path)
    x_np = aidpd.iq_array(x_c)
    y_np = aidpd.iq_array(y_c)
    split_idx = int(min(len(x_np), len(y_np)) * 0.8)
    x_train = x_np[:split_idx]
    y_train = y_np[:split_idx]
    x_val = x_np[split_idx:]
    y_val = y_np[split_idx:]
    g_opt = np.vdot(x_c[:split_idx], y_c[:split_idx]) / np.vdot(x_c[:split_idx], x_c[:split_idx])
    return x_train, y_train, x_val, y_val, float(g_opt.real), float(g_opt.imag)


def make_loader(x_np, y_np, frame_length, stride_length, batch_size):
    return DataLoader(
        aidpd.FramedDataset(x_np, y_np, frame_length, stride_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def fullseq_nmse(model, x_val, y_val):
    nmse, nmse_db = aidpd.evaluate_full_sequence_nmse(model, x_val, y_val)
    return float(nmse), float(nmse_db)


def compute_aclr_metrics(model, x_val, y_val, fs_mhz, main_band, lower_band, upper_band):
    model.eval()
    pred_iq = aidpd.run_sequence(model, x_val)
    target_iq = y_val.T
    nmse = np.sum((pred_iq - target_iq) ** 2) / np.sum(target_iq ** 2)
    pred_complex = aidpd.tensor_to_complex(pred_iq)
    freq_hz, pred_psd = aidpd.psd_db(pred_complex, fs_hz=fs_mhz * 1e6)
    pred_aclr = aidpd.aclr_from_psd(freq_hz, pred_psd, main_band, lower_band, upper_band)
    return {
        "nmse_linear": float(nmse),
        "nmse_db": float(10 * np.log10(nmse)),
        "aclr_lower": float(pred_aclr[0]),
        "aclr_upper": float(pred_aclr[1]),
        "aclr_worst": float(pred_aclr[2]),
    }


def channel_scores_from_state(source):
    src_h = int(source["in_conv.weight"].shape[0])
    scores = torch.zeros(src_h, dtype=torch.float64)
    for key in ("in_conv.weight", "attn.env_conv.weight"):
        if key in source:
            value = source[key].to(dtype=torch.float64)
            scores += torch.linalg.vector_norm(value, dim=tuple(range(1, value.ndim)))
    if "out_conv.weight" in source:
        scores += torch.linalg.vector_norm(source["out_conv.weight"].to(dtype=torch.float64), dim=(0, 2))
    for key, value in source.items():
        if key.startswith("tcn_blocks.") and key.endswith(".0.weight"):
            scores += torch.linalg.vector_norm(value.to(dtype=torch.float64), dim=(1, 2))
        if key.startswith("tcn_blocks.") and key.endswith(".1.weight") and value.ndim == 3:
            matrix = value[:, :, 0].to(dtype=torch.float64)
            if matrix.shape[0] == src_h and matrix.shape[1] == src_h:
                scores += torch.linalg.vector_norm(matrix, dim=0)
                scores += torch.linalg.vector_norm(matrix, dim=1)
    return scores


def copy_in_conv(source_weight, target_weight, keep):
    out = torch.zeros_like(target_weight)
    src = source_weight[keep].to(dtype=target_weight.dtype)
    if src.shape[1] == target_weight.shape[1]:
        out[:, :, :] = src[:, :, :]
        return out
    if src.shape[1] == 6 and target_weight.shape[1] == 9:
        out[:, 0:4, :] = src[:, 0:4, :]
        out[:, 6, :] = src[:, 4, :]
        out[:, 7, :] = src[:, 5, :]
        return out
    if src.shape[1] == 6 and target_weight.shape[1] == 11:
        out[:, 0:4, :] = src[:, 0:4, :]
        out[:, 8, :] = src[:, 4, :]
        out[:, 9, :] = src[:, 5, :]
        return out
    c = min(src.shape[1], target_weight.shape[1])
    out[:, :c, :] = src[:, :c, :]
    return out


def copy_attention(source_weight, target_weight, keep):
    out = torch.zeros_like(target_weight)
    src = source_weight[keep].to(dtype=target_weight.dtype)
    h = min(src.shape[0], out.shape[0])
    c = min(src.shape[1], out.shape[1])
    out[:h, :c, :] = src[:h, :c, :]
    return out


def copy_fir_center(source, target):
    out = target.clone()
    out.zero_()
    src_center = source.shape[0] // 2
    dst_center = target.shape[0] // 2
    half = min(src_center, dst_center, source.shape[0] - src_center - 1, target.shape[0] - dst_center - 1)
    out[dst_center - half: dst_center + half + 1] = source[
        src_center - half: src_center + half + 1
    ].to(dtype=target.dtype)
    return out


def copy_dense_to_rank1(source_dense, target_compress, target_expand, keep):
    dense = source_dense[keep][:, keep, 0].to(dtype=torch.float64)
    rank = target_compress.shape[0]
    compress = torch.zeros_like(target_compress)
    expand = torch.zeros_like(target_expand)
    try:
        u, s, vh = torch.linalg.svd(dense, full_matrices=False)
        use_rank = min(rank, s.numel())
        root_s = torch.sqrt(s[:use_rank])
        compress[:use_rank, :dense.shape[1], 0] = (root_s[:, None] * vh[:use_rank, :]).to(dtype=compress.dtype)
        expand[:dense.shape[0], :use_rank, 0] = (u[:, :use_rank] * root_s[None, :]).to(dtype=expand.dtype)
    except RuntimeError:
        use_rank = min(rank, dense.shape[0], dense.shape[1])
        compress[:use_rank, :use_rank, 0] = torch.eye(use_rank, dtype=compress.dtype)
        expand[:dense.shape[0], :use_rank, 0] = dense[:, :use_rank].to(dtype=expand.dtype)
    return compress, expand


def load_pruned_994(model, checkpoint_path):
    source = aidpd.load_state_dict(checkpoint_path)
    target = model.state_dict()
    src_h = int(source["in_conv.weight"].shape[0])
    dst_h = int(target["in_conv.weight"].shape[0])
    if dst_h > src_h:
        raise ValueError("Pruned init only supports destination hidden <= source hidden.")
    keep = torch.topk(channel_scores_from_state(source), dst_h).indices.sort().values
    keep_list = [int(x) for x in keep.tolist()]
    new_state = {key: value.clone() for key, value in target.items()}

    for key in ["g_real", "g_imag", "g3_real", "g3_imag", "g5_real", "g5_imag"]:
        if key in source and key in new_state:
            new_state[key] = source[key].to(dtype=new_state[key].dtype)
    for key in list(new_state):
        if key.startswith("extra_poly_") and key in source:
            new_state[key] = source[key].to(dtype=new_state[key].dtype)

    new_state["in_conv.weight"] = copy_in_conv(source["in_conv.weight"], new_state["in_conv.weight"], keep)
    new_state["attn.env_conv.weight"] = copy_attention(
        source["attn.env_conv.weight"], new_state["attn.env_conv.weight"], keep
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
        if block_id not in src_blocks:
            continue
        new_state[depth_key] = source[depth_key][keep, :, :].to(dtype=new_state[depth_key].dtype)
        dense_key = f"tcn_blocks.{block_id}.1.weight"
        if dense_key in source and dense_key in new_state:
            if new_state[dense_key].shape[0] == dst_h:
                new_state[dense_key] = source[dense_key][keep][:, keep, :].to(dtype=new_state[dense_key].dtype)
            else:
                expand_key = f"tcn_blocks.{block_id}.2.weight"
                compress, expand = copy_dense_to_rank1(
                    source[dense_key], new_state[dense_key], new_state[expand_key], keep
                )
                new_state[dense_key] = compress
                new_state[expand_key] = expand

    if "fir_real" in new_state and "fir_real" in source:
        new_state["fir_real"] = copy_fir_center(source["fir_real"], new_state["fir_real"])
        new_state["fir_imag"] = copy_fir_center(source["fir_imag"], new_state["fir_imag"])

    model.load_state_dict(new_state, strict=True)
    return keep_list


def fit_fir(model, x_train, y_train, args):
    if args.fir_fit_domain == "time":
        return aidpd.fit_output_fir_from_train(model, x_train, y_train)
    return aidpd.fit_output_fir_freq_from_train(
        model,
        x_train,
        y_train,
        fs_mhz=args.fs_mhz,
        main_band=tuple(args.main_band),
        lower_band=tuple(args.lower_band),
        upper_band=tuple(args.upper_band),
        main_weight=args.fir_main_weight,
        adjacent_weight=args.fir_adjacent_weight,
        floor_weight=args.fir_floor_weight,
        ridge=args.fir_ridge,
    )


def set_fir_trainable(model, trainable):
    for name, param in model.named_parameters():
        if name.startswith("fir_"):
            param.requires_grad = trainable


def score_metrics(metrics, aclr_target):
    aclr_penalty = max(0.0, metrics["aclr_worst"] - aclr_target)
    return metrics["nmse_db"] + 0.35 * aclr_penalty


def train_model(model, x_train, y_train, x_val, y_val, checkpoint_path, args):
    train_loader = make_loader(x_train, y_train, args.frame_length, args.stride_length, args.batch_size)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    best_score = float("inf")
    best_metrics = None
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            pred = model(batch_x)
            loss = aidpd.loss_with_optional_terms(
                pred,
                batch_y,
                crop_margin=args.crop_margin,
                diff_loss_weight=args.diff_loss_weight,
                spectral_loss_weight=args.spectral_loss_weight,
                fs_mhz=args.fs_mhz,
                main_band=tuple(args.main_band),
                lower_band=tuple(args.lower_band),
                upper_band=tuple(args.upper_band),
                spectral_main_weight=args.spectral_main_weight,
                spectral_adjacent_weight=args.spectral_adjacent_weight,
                spectral_floor_weight=args.spectral_floor_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        nmse, nmse_db = fullseq_nmse(model, x_val, y_val)
        scheduler.step(nmse_db)
        history.append((epoch, nmse_db))
        if epoch % args.refit_every == 0:
            fit_fir(model, x_train, y_train, args)

        if epoch % args.aclr_every == 0 or epoch == args.epochs:
            metrics = compute_aclr_metrics(
                model,
                x_val,
                y_val,
                args.fs_mhz,
                tuple(args.main_band),
                tuple(args.lower_band),
                tuple(args.upper_band),
            )
            current_score = score_metrics(metrics, args.aclr_target_dbc)
            if current_score < best_score:
                best_score = current_score
                best_metrics = metrics
                torch.save(model.state_dict(), checkpoint_path)
            aclr_text = f" | ACLR {metrics['aclr_worst']:.2f} dBc | score {current_score:.2f}"
        else:
            aclr_text = ""

        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"Full NMSE {nmse_db:.2f} dB{aclr_text}"
        )

    if best_metrics is None:
        best_metrics = compute_aclr_metrics(
            model,
            x_val,
            y_val,
            args.fs_mhz,
            tuple(args.main_band),
            tuple(args.lower_band),
            tuple(args.upper_band),
        )
        torch.save(model.state_dict(), checkpoint_path)
    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    return history, best_metrics


def plot_history(history, out_path, title):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    plt.plot([x for x, _ in history], [y for _, y in history], label="full-sequence NMSE")
    plt.axhline(-35.0, color="black", linestyle="--", linewidth=1.0, label="-35 dB")
    plt.axhline(-37.0, color="tab:red", linestyle=":", linewidth=1.0, label="-37 dB")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE (dB)")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def build_model(config, g_real, g_imag):
    model = aidpd.DiagramDPDNet(g_real, g_imag, config=config).to(DEVICE)
    multipliers = aidpd.real_multiplier_count_from_config(config)
    if multipliers > 800:
        raise RuntimeError(f"Multiplier budget exceeded: {multipliers} > 800 for {config}")
    return model, multipliers


def initialize_model(model, init_mode, init_checkpoint):
    if init_mode == "none" or init_checkpoint is None or not init_checkpoint.exists():
        print(">>> Fresh initialization.")
        return
    if init_mode == "full":
        model.load_state_dict(aidpd.load_state_dict(init_checkpoint), strict=True)
        print(f">>> Full init from {init_checkpoint}")
        return
    if init_mode == "prune994":
        keep = load_pruned_994(model, init_checkpoint)
        print(f">>> Pruned init from {init_checkpoint}")
        print(f">>> Kept source channels: {keep}")
        return
    loaded, partial, skipped = aidpd.load_partial_state(model, init_checkpoint)
    print(f">>> Partial init from {init_checkpoint}")
    print(f">>> Loaded tensors: exact={len(loaded)}, partial={len(partial)}, skipped={len(skipped)}")


def run_variant(name, args, x_train, y_train, x_val, y_val, g_real, g_imag):
    variant = VARIANTS[name]
    config = dict(variant["config"])
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else CHECKPOINT_DIR / f"TCN_800_{name}_best.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model, multipliers = build_model(config, g_real, g_imag)

    print("\n" + "=" * 88)
    print(f">>> Variant: {name}")
    print(f">>> Description: {variant['description']}")
    print(f">>> Config: {config}")
    print(f">>> Real multipliers: {multipliers}")
    initialize_model(model, args.init_mode or variant.get("init", "partial"), resolve_path(args.init_checkpoint))

    if args.freeze_fir:
        set_fir_trainable(model, False)
        print(">>> Output FIR frozen during gradient training.")

    print(f">>> Initial {args.fir_fit_domain}-domain train-only FIR fit.")
    fit_fir(model, x_train, y_train, args)
    init_metrics = compute_aclr_metrics(
        model,
        x_val,
        y_val,
        args.fs_mhz,
        tuple(args.main_band),
        tuple(args.lower_band),
        tuple(args.upper_band),
    )
    print(
        f">>> After initial FIR | NMSE {init_metrics['nmse_db']:.2f} dB | "
        f"ACLR {init_metrics['aclr_worst']:.2f} dBc"
    )
    torch.save(model.state_dict(), checkpoint_path)

    if args.evaluate_initial_only or args.epochs <= 0:
        final_metrics = init_metrics
        history = [(0, init_metrics["nmse_db"])]
    else:
        history, _ = train_model(model, x_train, y_train, x_val, y_val, checkpoint_path, args)

        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
        print(f">>> Final {args.fir_fit_domain}-domain train-only FIR refit.")
        fit_fir(model, x_train, y_train, args)
        torch.save(model.state_dict(), checkpoint_path)
        final_metrics = compute_aclr_metrics(
            model,
            x_val,
            y_val,
            args.fs_mhz,
            tuple(args.main_band),
            tuple(args.lower_band),
            tuple(args.upper_band),
        )

    convergence_path = FIGURE_DIR / f"{checkpoint_path.stem}_convergence.png"
    plot_history(history, convergence_path, f"TCN 800 {name} ({multipliers} multipliers)")

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

    print(
        f">>> FINAL {name}: multipliers={multipliers} | "
        f"NMSE {final_metrics['nmse_db']:.2f} dB | ACLR {final_metrics['aclr_worst']:.2f} dBc"
    )
    print(f">>> Saved checkpoint: {checkpoint_path}")
    return {
        "variant": name,
        "multipliers": multipliers,
        "nmse_db": final_metrics["nmse_db"],
        "aclr_worst": final_metrics["aclr_worst"],
        "checkpoint": str(checkpoint_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="<=800 real-multiplier ACLR-oriented TCN/EA-TCN experiments.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="h10_base6_l3_fir41")
    parser.add_argument("--sweep", action="store_true", help="Run all variants sequentially.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT))
    parser.add_argument("--init-mode", choices=["prune994", "partial", "full", "none"], default=None)
    parser.add_argument("--evaluate-initial-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-length", type=int, default=2048)
    parser.add_argument("--stride-length", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=12)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--freeze-fir", action="store_true")
    parser.add_argument("--refit-every", type=int, default=20)
    parser.add_argument("--aclr-every", type=int, default=10)
    parser.add_argument("--crop-margin", type=int, default=128)
    parser.add_argument("--diff-loss-weight", type=float, default=0.02)
    parser.add_argument("--spectral-loss-weight", type=float, default=2e-4)
    parser.add_argument("--spectral-main-weight", type=float, default=1.0)
    parser.add_argument("--spectral-adjacent-weight", type=float, default=8.0)
    parser.add_argument("--spectral-floor-weight", type=float, default=0.05)
    parser.add_argument("--fir-fit-domain", choices=["time", "freq"], default="time")
    parser.add_argument("--fir-main-weight", type=float, default=1.0)
    parser.add_argument("--fir-adjacent-weight", type=float, default=6.0)
    parser.add_argument("--fir-floor-weight", type=float, default=0.05)
    parser.add_argument("--fir-ridge", type=float, default=1e-7)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--main-band", type=float, nargs=2, default=(-100.0, 100.0))
    parser.add_argument("--lower-band", type=float, nargs=2, default=(-245.0, -180.0))
    parser.add_argument("--upper-band", type=float, nargs=2, default=(180.0, 245.0))
    parser.add_argument("--aclr-target-dbc", type=float, default=-45.0)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data_path = resolve_path(args.data)
    x_train, y_train, x_val, y_val, g_real, g_imag = load_data(data_path)
    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Data: {data_path}")
    print(f">>> Train/val samples: {len(x_train)} / {len(x_val)}")

    variants = sorted(VARIANTS) if args.sweep else [args.variant]
    results = []
    for variant in variants:
        results.append(run_variant(variant, args, x_train, y_train, x_val, y_val, g_real, g_imag))

    print("\n>>> 800-MAC summary")
    for item in sorted(results, key=lambda x: (x["aclr_worst"] > args.aclr_target_dbc, x["nmse_db"])):
        print(
            f"{item['variant']}: {item['multipliers']} MAC | "
            f"NMSE {item['nmse_db']:.2f} dB | ACLR {item['aclr_worst']:.2f} dBc | "
            f"{item['checkpoint']}"
        )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
