# Lowest-point ACLR results

This is now the only ACLR definition used by the final evaluator and optimizer:

`ACLR = lowest adjacent-band error PSD point - lowest main-band dataset-y PSD point`.

The main band is 20--80 MHz.  The adjacent bands are -180---120 MHz and
120--180 MHz.  `error = target_y - pred_y`; the reported EA-TCN ACLR is the worse
(larger) of the lower/upper results.  Welch uses 4096 points.

| Model | Selected starting point | Final NMSE dB | Final lower dBc | Final upper dBc | Final worst dBc | Gain from selected start |
|---|---:|---:|---:|---:|---:|---:|
| 2284 | -45.259 | -40.063 | -45.305 | -45.307 | **-45.305** | 0.046 dB |
| 998 | -42.255 | -37.004 | -42.660 | -42.937 | **-42.660** | 0.405 dB |
| 786 | -40.858 | -35.069 | -41.522 | -42.337 | **-41.522** | 0.664 dB |
| 796 | -40.771 | -35.188 | -40.842 | -41.079 | **-40.842** | 0.072 dB |

The dataset reference-error worst ACLR is -45.105 dBc.  Model 2284 beats it by
0.200 dB.  Models 998, 786 and 796 remain 2.444, 3.582 and 4.262 dB above it.

Training uses an annealed soft-min only as a differentiable surrogate.  Epoch
acceptance, rollback, checkpoint selection and weight extrapolation all use the
true hard minimum-point ACLR plus the model-specific NMSE constraint.

The PSD plot contains only four curves: Dataset y, Predicted y, Reference error,
and EA-TCN error (`target y - predicted y`).  It contains no band shading, ACLR
threshold line or title.

## Final checkpoints

- 2284: `../aclr_min_point_extrapolation/scan_2284/2284/best.pth`
- 998: `../aclr_min_point_extrapolation/scan_998/998/best.pth`
- 786: `../aclr_min_point_extrapolation/scan_786_wide/786/best.pth`
- 796: `../aclr_min_point_extrapolation/scan_796_wide/796/best.pth`

## Reproducible scripts

- `steps/aclr_min_point_optimization.py`
- `steps/aclr_min_point_extrapolation.py`
- `steps/final_evaluate_unified.py`
- `tools/export_matlab_psd_data.py`
- `eval_outputs_direct/matlab_psd/plot_model_psd.m`
