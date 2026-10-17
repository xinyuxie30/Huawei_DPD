# TCN 999-MAC experiment summary - 2026-06-28

## Scope

Goal: keep real multipliers <= 999 and search for the best NMSE/ACLR tradeoff on the 20% validation split.

Evaluation bands:
- Main band: [-100, 100] MHz
- Lower adjacent band: [-245, -180] MHz
- Upper adjacent band: [180, 245] MHz
- ACLR target: worst ACLR <= -45 dBc

Input feature notation:
- Let `x[n] = I[n] + jQ[n]` and `r[n] = |x[n]|`.
- `Input-6 = [I[n], Q[n], I[n+1], Q[n+1], r[n], r[n]^3]`.
- `Input-9 = [I[n], Q[n], I[n+1], Q[n+1], I[n-1], Q[n-1], r[n], r[n]^3, r[n]^5]`.

## Script changes

- `steps/AI-DPD.py`
  - Added preset `rich9_attn3_fir22_992`.
  - This is the <=999 equivalent of `rich9_attn3_fir23_1000`, with output FIR reduced from 47 taps to 45 taps.

- `steps/step40.1.6_memory_fir1000.py`
  - Added runtime controls for `hidden`, `input-mode`, `attention-channels`, `output-fir-taps`, and `max-multipliers`.
  - Added frequency-domain output FIR refit controls:
    - `--fir-fit-domain`
    - `--fir-main-weight`
    - `--fir-adjacent-weight`
    - `--fir-floor-weight`
    - `--fir-ridge`
  - Default behavior remains the original 1000-MAC script.

## Tested candidates

| Candidate | Real mults | Key design | Final FIR fit | Val NMSE | Worst ACLR | Result |
| --- | ---: | --- | --- | ---: | ---: | --- |
| TCN_999_memory_fir45_992 | 992 | Input-9, hidden=10, L4, attention=3, FIR=45 | time LS | -37.20 dB | -46.59 dBc | Pass, strong ACLR |
| TCN_999_base6_attn3_fir53_994 | 994 | Input-6, hidden=10, L4, attention=3, FIR=53 | time LS | -36.91 dB | -48.00 dBc | ACLR strong, NMSE weaker |
| TCN_999_rich9_attn4_fir39_998 | 998 | Input-9, hidden=10, L4, attention=4, FIR=39 | time LS | -37.39 dB | -44.75 dBc | NMSE best, ACLR fail |
| TCN_999_rich9_attn4_fir39_998_freqfit_w0p03_floor0p04 | 998 | Input-9, hidden=10, L4, attention=4, FIR=39 | freq LS, adjacent=0.03, floor=0.04 | -37.37 dB | -45.30 dBc | Recommended robust |
| TCN_999_rich9_attn4_fir39_998_freqfit_w0p03_floor0p05 | 998 | Input-9, hidden=10, L4, attention=4, FIR=39 | freq LS, adjacent=0.03, floor=0.05 | -37.37 dB | -45.04 dBc | Best NMSE pass, very small ACLR margin |

## Recommended checkpoint

Robust recommended checkpoint:

```text
checkpoints/AI-DPD/TCN_999_rich9_attn4_fir39_998_freqfit_w0p03_floor0p04.pth
```

Metrics:

```text
Real multipliers: 998
Linear NMSE: 1.833816e-04
dB NMSE: -37.37 dB
Worst ACLR: -45.30 dBc
ACLR margin to -45 dBc: 0.30 dB
```

Figures:

```text
eval_outputs_direct/figures/TCN_999_rich9_attn4_fir39_998_freqfit_w0p03_floor0p04_memory4_fir19_998_psd_aclr.png
eval_outputs_direct/figures/TCN_999_rich9_attn4_fir39_998_freqfit_w0p03_floor0p04_memory4_fir19_998_aclr_bar.png
eval_outputs_direct/figures/TCN_999_rich9_attn4_fir39_998_best_convergence.png
```

## Reproduction command

This command trains the <=999 model from the SWA baseline and applies the final frequency-domain FIR refit:

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\steps\step40.1.6_memory_fir1000.py `
  --input-mode rich9 `
  --attention-channels 4 `
  --output-fir-taps 39 `
  --max-multipliers 999 `
  --checkpoint .\checkpoints\AI-DPD\TCN_999_rich9_attn4_fir39_998_final_retrain.pth `
  --epochs-base 600 `
  --epochs-long 240 `
  --base-lr 5e-3 `
  --long-lr 5e-6 `
  --batch-size 16 `
  --long-batch-size 4 `
  --frame-length 800 `
  --stride-length 32 `
  --long-frame-length 4096 `
  --long-stride-length 256 `
  --target-nmse-db -37.25 `
  --fir-fit-domain freq `
  --fir-adjacent-weight 0.03 `
  --fir-floor-weight 0.04
```

## Interpretation

- Input-9 is necessary for the best NMSE under this budget. Input-6 with a longer FIR gives better ACLR but loses about 0.46 dB NMSE.
- Increasing attention channels from 3 to 4 improves nonlinear/memory fitting, but the shorter FIR weakens ACLR if the final FIR is refit by time-domain LS only.
- Frequency-domain FIR refit is the practical fix: it recovers ACLR while keeping most of the NMSE gain from the 4-channel attention model.
