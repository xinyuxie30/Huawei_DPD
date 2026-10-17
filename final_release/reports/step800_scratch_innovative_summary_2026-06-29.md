# <=800 MAC scratch-only innovative search - 2026-06-29

## Goal

Hard constraint: real multipliers < 800.

Training constraint: student starts from random initialization and uses dataset `target_y` directly. No checkpoint initialization and no teacher distillation target.

Target:

- Validation NMSE <= -40 dB
- Worst ACLR <= -45 dBc

Input feature notation:

- Let `x[n] = I[n] + jQ[n]` and `r[n] = |x[n]|`.
- `Input-6 = [I[n], Q[n], I[n+1], Q[n+1], r[n], r[n]^3]`.
- `Input-7 = [I[n], Q[n], I[n+1], Q[n+1], r[n], r[n]^3, r[n]^5]`.
- `Input-8 = [I[n], Q[n], I[n+1], Q[n+1], I[n-1], Q[n-1], r[n], r[n]^3]`.

## New script

```text
steps/step800_scratch_innovative.py
```

This script does not modify the final report scripts. It imports `final_train_lightweight_999_distill.py`, registers experimental sub-800 variants, and forces:

```text
--teacher-mode none
--distill-loss-weight 0
--distill-diff-weight 0
```

The main idea is to keep output FIR capacity and use low-rank pointwise TCN layers to add depth under the 800-MAC budget.

## Candidate pool

| Variant | Key design | Real multipliers |
| --- | --- | ---: |
| `h10_l4_base6_rank3_fir41` | H10/L4, Input-6, pointwise rank 3, FIR41 | 786 |
| `h10_l4_semirich7_rank3_fir39` | H10/L4, Input-7, pointwise rank 3, FIR39 | 788 |
| `h10_l4_semirich8_rank3_fir37` | H10/L4, Input-8, pointwise rank 3, FIR37 | 790 |
| `h11_l4_base6_rank2_fir47` | H11/L4, Input-6, pointwise rank 2, FIR47 | 783 |
| `h12_l5_base6_rank1_fir39` | H12/L5, Input-6, pointwise rank 1, FIR39 | 792 |
| `h13_l5_base6_rank1_fir27` | H13/L5, Input-6, pointwise rank 1, FIR27 | 796 |

## Probe runs completed

The following runs used a medium probe schedule rather than the full default schedule:

```text
--stage1-epochs 60
--stage2-epochs 30
--stage3-epochs 30
```

| Variant | Real multipliers | NMSE | ACLR | Result |
| --- | ---: | ---: | ---: | --- |
| `h10_l4_base6_rank3_fir41` | 786 | -34.83 dB | -46.77 dBc | ACLR pass, NMSE below current 796-MAC baseline |
| `h11_l4_base6_rank2_fir47` | 783 | -33.58 dB | -46.74 dBc | ACLR pass, rank 2 bottleneck too narrow |
| `h10_l4_semirich8_rank3_fir37` | 790 | -34.59 dB | -44.76 dBc | NMSE slight recovery, ACLR fail |

Reference baseline:

| Model | Real multipliers | NMSE | ACLR |
| --- | ---: | ---: | ---: |
| `final_lightweight_796_H10_L3_FIR41_scratch_retrain` | 796 | -35.20 dB | -47.03 dBc |

## Interpretation

The probe results show that adding depth under 800 MAC is not enough. The available budget forces one of three harmful tradeoffs:

1. Reduce full pointwise TCN mixing to a low-rank bottleneck.
2. Shorten output FIR.
3. Reduce hidden width or remove useful input features.

The data currently indicates:

- Output FIR length is essential for ACLR.
- Full pointwise mixing is essential for NMSE.
- Extra channels beyond Input-6 only help if FIR is not shortened too much.
- Rank 1 or rank 2 bottlenecks are too restrictive for this dataset.

## Current conclusion

With the tested scratch-only sub-800 architectures, `-40 dB` validation NMSE is not reachable. The best known sub-800 result remains:

```text
796 MAC | NMSE -35.20 dB | ACLR -47.03 dBc
```

The gap to `-40 dB` is about 4.6 dB, which is too large to plausibly close by learning-rate tuning alone.

## Remaining commands if more evidence is needed

Run the two not-yet-probed variants:

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\steps\step800_scratch_innovative.py --variant h10_l4_semirich7_rank3_fir39
& D:\anconda\envs\pytorch\python.exe -B .\steps\step800_scratch_innovative.py --variant h12_l5_base6_rank1_fir39
& D:\anconda\envs\pytorch\python.exe -B .\steps\step800_scratch_innovative.py --variant h13_l5_base6_rank1_fir27
```

Recommended full run among the tested variants:

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\steps\step800_scratch_innovative.py --variant h10_l4_base6_rank3_fir41
```

This is the only tested low-rank depth variant that preserves ACLR while staying close to the 796-MAC baseline.
