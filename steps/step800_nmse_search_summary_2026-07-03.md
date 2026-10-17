# Step800 NMSE Search Summary - 2026-07-03

Constraint:

- Real multipliers <= 800.
- Keep the existing NMSE/ACLR evaluation settings unchanged.
- Store new attempt scripts, checkpoints, figures, and unified-eval outputs under `steps/`.

Input feature notation:

- Let `x[n] = I[n] + jQ[n]` and `r[n] = |x[n]|`.
- `Input-6 = [I[n], Q[n], I[n+1], Q[n+1], r[n], r[n]^3]`.
- `Input-9 = [I[n], Q[n], I[n+1], Q[n+1], I[n-1], Q[n-1], r[n], r[n]^3, r[n]^5]`.
- `Input-11 = [I[n], Q[n], I[n+1], Q[n+1], I[n-1], Q[n-1], I[n]-I[n-1], Q[n]-Q[n-1], r[n], r[n]^3, r[n]^5]`.

New launcher:

```text
steps/step800_nmse_search.py
```

Artifact root:

```text
steps/step800_nmse_search/
```

Unified evaluation output:

```text
steps/step800_nmse_search/unified_eval/summary.csv
```

## Final Unified-Eval Results

| Checkpoint | MACs | NMSE dB | Worst ACLR dBc | Result |
| --- | ---: | ---: | ---: | --- |
| `h10_l3_base6_poly7_fir39_792_distill.pth` | 792 | -35.67 | -45.34 | Best new valid result |
| `step800_scratch_h10_l4_base6_rank3_fir41_786.pth` | 786 | -35.46 | -46.76 | Previous best from user run |
| `final_lightweight_796_H10_L3_FIR41_scratch_retrain.pth` | 796 | -35.20 | -47.03 | Previous 796-MAC baseline |
| `h10_l3_rich9_fir33_794_distill.pth` | 794 | -35.30 | -43.85 | ACLR fail |
| `h9_l3_rich9_fir57_798_distill.pth` | 798 | -34.56 | -47.98 | ACLR pass, NMSE worse |
| `h9_l3_rich11_fir53_800_distill.pth` | 800 | -34.83 | -46.97 | ACLR pass, NMSE worse |
| `h10_l4_rich9_rank3_fir33_784_distill.pth` | 784 | -34.17 | -43.71 | Low-rank bottleneck and ACLR fail |
| `h9_l4_rich9_fir25_796_distill.pth` | 796 | -34.80 | -38.38 | ACLR fail |

## Best Checkpoint

```text
steps/step800_nmse_search/checkpoints/h10_l3_base6_poly7_fir39_792_distill.pth
```

Unified metrics:

```text
Real multipliers: 792
Linear NMSE: 2.711606e-04
dB NMSE: -35.67 dB
Worst ACLR: -45.34 dBc
Target: NMSE <= -40.00 dB, ACLR <= -45.00 dBc
```

## Interpretation

The best valid new result improves the latest 786-MAC result by about 0.21 dB
NMSE while still passing ACLR, but it remains about 4.33 dB short of the -40 dB
NMSE target.

The tested tradeoffs show the current <=800-MAC limit is dominated by backbone
capacity:

- H10/L3 with Input-6 and full pointwise mixing remains the strongest family.
- Adding a 7th-order polynomial branch helps slightly if FIR is shortened from
  41 to 39 taps.
- Input-9 and Input-11 under 800 MAC require either narrower hidden width or
  shorter FIR, and both hurt the final score.
- L4 depth with rank bottlenecks is not competitive for NMSE.
- Short-FIR Input-9 variants can train but fail ACLR badly under the unchanged
  evaluator.
