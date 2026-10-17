import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AI_DPD_PATH = SCRIPT_DIR / "AI-DPD.py"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
RESULT_DIR = PROJECT_ROOT / "eval_outputs_direct" / "blend_routes"
DEFAULT_STUDENT = CHECKPOINT_DIR / "EA-TCN_memory_fir1000_best.pth"
DEFAULT_TEACHER = CHECKPOINT_DIR / "teacher32_6layer_scratch_best.pth"

STUDENT_CONFIG = {
    "hidden": 10,
    "input_mode": "rich9",
    "dilations": [1, 2, 4, 8],
    "attention_channels": 3,
    "poly_orders": [1, 3, 5],
    "output_fir_taps": 47,
}


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd", AI_DPD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd_module()
DEVICE = aidpd.DEVICE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Teacher-student output blend with train-only complex FIR calibration."
    )
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--student-checkpoint", default=str(DEFAULT_STUDENT))
    parser.add_argument("--teacher-checkpoint", default=str(DEFAULT_TEACHER))
    parser.add_argument("--alpha", type=float, default=None, help="Fixed teacher blend weight. If omitted, grid search.")
    parser.add_argument("--alpha-grid", type=float, nargs=3, default=(0.0, 1.0, 0.05))
    parser.add_argument("--fir-taps", type=int, nargs="+", default=[31, 47, 63])
    parser.add_argument("--aclr-target-dbc", type=float, default=-45.0)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--main-band", type=float, nargs=2, default=(-100.0, 100.0))
    parser.add_argument("--lower-band", type=float, nargs=2, default=(-245.0, -180.0))
    parser.add_argument("--upper-band", type=float, nargs=2, default=(180.0, 245.0))
    parser.add_argument("--output-prefix", default="teacher_student_fir_blend")
    return parser.parse_args()


def resolve_path(path_like):
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
    g_opt = np.vdot(x_c[:split_idx], y_c[:split_idx]) / np.vdot(x_c[:split_idx], x_c[:split_idx])
    return (
        x_np[:split_idx],
        y_np[:split_idx],
        x_np[split_idx:],
        y_np[split_idx:],
        float(g_opt.real),
        float(g_opt.imag),
    )


def run_model(model, x_np):
    model.eval()
    x_tensor = torch.as_tensor(x_np.T, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return model(x_tensor).squeeze(0).cpu().numpy().T


def iq_to_complex(iq):
    return iq[:, 0] + 1j * iq[:, 1]


def complex_to_iq(signal):
    return np.stack([signal.real, signal.imag], axis=1)


def shifted_columns(signal, taps):
    if taps % 2 == 0:
        raise ValueError("FIR taps must be odd for centered same-length calibration.")
    pad = taps // 2
    columns = []
    for tap in range(taps):
        shift = tap - pad
        if shift < 0:
            col = np.r_[signal[-shift:], np.repeat(signal[-1], -shift)]
        elif shift > 0:
            col = np.r_[np.repeat(signal[0], shift), signal[:-shift]]
        else:
            col = signal
        columns.append(col)
    return np.stack(columns, axis=1)


def fit_complex_fir(pred_train, target_train, taps):
    coeffs = np.linalg.lstsq(shifted_columns(pred_train, taps), target_train, rcond=None)[0]
    return coeffs


def apply_complex_fir(pred, coeffs):
    return shifted_columns(pred, len(coeffs)) @ coeffs


def nmse(pred_iq, target_iq):
    value = np.sum((pred_iq - target_iq) ** 2) / np.sum(target_iq ** 2)
    return float(value), float(10 * np.log10(value))


def aclr(pred_iq, args):
    freq_hz, psd = aidpd.psd_db(iq_to_complex(pred_iq), fs_hz=args.fs_mhz * 1e6)
    lower, upper, worst = aidpd.aclr_from_psd(
        freq_hz,
        psd,
        tuple(args.main_band),
        tuple(args.lower_band),
        tuple(args.upper_band),
    )
    return float(lower), float(upper), float(worst)


def alpha_values(args):
    if args.alpha is not None:
        return [float(args.alpha)]
    start, stop, step = args.alpha_grid
    count = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 10) for i in range(count)]


def main():
    args = parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = resolve_path(args.data)
    student_path = resolve_path(args.student_checkpoint)
    teacher_path = resolve_path(args.teacher_checkpoint)

    x_train, y_train, x_val, y_val, g_real, g_imag = load_data(data_path)
    if not student_path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {student_path}")
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

    student = aidpd.DiagramDPDNet(g_real, g_imag, config=STUDENT_CONFIG).to(DEVICE)
    student.load_state_dict(aidpd.load_state_dict(student_path), strict=True)
    teacher = aidpd.DiagramDPDNet(g_real, g_imag, config=aidpd.preset_config("teacher32_6layer")).to(DEVICE)
    teacher.load_state_dict(aidpd.load_state_dict(teacher_path), strict=True)

    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Student: {student_path} | MACs={aidpd.real_multiplier_count_from_config(STUDENT_CONFIG)}")
    print(
        f">>> Teacher: {teacher_path} | "
        f"MACs={aidpd.real_multiplier_count_from_config(aidpd.preset_config('teacher32_6layer'))}"
    )
    print(">>> This route is a non-lightweight target/diagnostic model, not the final <=1000-MAC submission.")

    student_train = run_model(student, x_train)
    student_val = run_model(student, x_val)
    teacher_train = run_model(teacher, x_train)
    teacher_val = run_model(teacher, x_val)

    rows = []
    best = None
    for alpha in alpha_values(args):
        blend_train = (1.0 - alpha) * student_train + alpha * teacher_train
        blend_val = (1.0 - alpha) * student_val + alpha * teacher_val
        for taps in args.fir_taps:
            coeffs = fit_complex_fir(iq_to_complex(blend_train), iq_to_complex(y_train), taps)
            pred_val = complex_to_iq(apply_complex_fir(iq_to_complex(blend_val), coeffs))
            linear_nmse, db_nmse = nmse(pred_val, y_val)
            lower, upper, worst = aclr(pred_val, args)
            row = {
                "alpha": alpha,
                "fir_taps": taps,
                "linear_nmse": linear_nmse,
                "nmse_db": db_nmse,
                "aclr_lower": lower,
                "aclr_upper": upper,
                "aclr_worst": worst,
                "coeffs": coeffs,
                "pred_val": pred_val,
            }
            rows.append(row)
            meets_aclr = worst <= args.aclr_target_dbc
            print(
                f"alpha={alpha:.2f} taps={taps:2d} | "
                f"NMSE {db_nmse:.2f} dB | ACLR worst {worst:.2f} dBc | "
                f"{'ACLR PASS' if meets_aclr else 'ACLR FAIL'}"
            )
            if meets_aclr and (best is None or db_nmse < best["nmse_db"]):
                best = row

    if best is None:
        best = min(rows, key=lambda item: item["nmse_db"])
        print("\n>>> No candidate met ACLR target; reporting best NMSE candidate.")
    else:
        print("\n>>> Best candidate satisfying ACLR target:")

    result_path = RESULT_DIR / f"{args.output_prefix}_best.npz"
    np.savez(
        result_path,
        alpha=best["alpha"],
        fir_taps=best["fir_taps"],
        fir_coeffs=best["coeffs"],
        nmse_db=best["nmse_db"],
        linear_nmse=best["linear_nmse"],
        aclr_lower=best["aclr_lower"],
        aclr_upper=best["aclr_upper"],
        aclr_worst=best["aclr_worst"],
        student_checkpoint=str(student_path),
        teacher_checkpoint=str(teacher_path),
    )
    print(
        f"alpha={best['alpha']:.2f} taps={best['fir_taps']} | "
        f"linear NMSE {best['linear_nmse']:.6e} | NMSE {best['nmse_db']:.2f} dB | "
        f"ACLR worst {best['aclr_worst']:.2f} dBc"
    )
    print(f">>> Saved blend coefficients: {result_path}")

    metrics = aidpd.save_psd_and_aclr_figures(
        y_val.T,
        best["pred_val"].T,
        output_prefix=args.output_prefix,
        main_band=tuple(args.main_band),
        lower_band=tuple(args.lower_band),
        upper_band=tuple(args.upper_band),
        fs_hz=args.fs_mhz * 1e6,
        target_dbc=args.aclr_target_dbc,
    )
    print(f">>> PSD figure: {metrics['psd_path']}")
    print(f">>> ACLR figure: {metrics['aclr_path']}")

    plt.figure(figsize=(9, 5))
    pass_rows = [row for row in rows if row["aclr_worst"] <= args.aclr_target_dbc]
    plot_rows = pass_rows if pass_rows else rows
    plt.scatter([row["alpha"] for row in plot_rows], [row["nmse_db"] for row in plot_rows], c=[row["fir_taps"] for row in plot_rows])
    plt.colorbar(label="FIR taps")
    plt.axhline(-40.0, color="tab:red", linestyle=":", linewidth=1.1, label="-40 dB")
    plt.xlabel("Teacher blend weight alpha")
    plt.ylabel("Validation NMSE (dB)")
    plt.title("Teacher-student FIR blend search")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    search_path = FIGURE_DIR / f"{args.output_prefix}_search.png"
    plt.savefig(search_path, dpi=300)
    plt.close()
    print(f">>> Search figure: {search_path}")


if __name__ == "__main__":
    main()
