# Three-metric ACLR experiment conclusion

All values use `error = target_y - pred_y`, validation split 80/20, Welch PSD,
main band 20--80 MHz, adjacent bands -180---120/120--180 MHz, and the fixed
5 MHz narrow bands 31.64--36.64/-129.84---124.84/121.08--126.08 MHz.

## Reference and best constraint-satisfying candidates

| Candidate | NMSE dB | Full mean dBc | Minimum point dBc | Narrow dBc |
|---|---:|---:|---:|---:|
| Dataset reference error | -36.46 | -43.74 | -45.10 | -45.00 |
| 2284 baseline | -40.15 | -42.98 | -44.72 | -43.83 |
| 2284 best experiment | -40.01 | -43.09 | -45.24 | -44.01 |
| 998 baseline | -37.37 | -40.09 | -41.97 | -41.38 |
| 998 best experiment | -37.00 | -40.32 | -42.26 | -41.74 |
| 786 baseline | -35.46 | -38.51 | -40.27 | -39.15 |
| 786 best experiment | -35.03 | -38.67 | -40.86 | -39.45 |
| 796 baseline | -35.20 | -38.03 | -40.18 | -38.69 |
| 796 best experiment | -35.17 | -38.17 | -40.77 | -38.91 |
| H32 ACLR teacher | -40.03 | -43.87 | -46.40 | -45.34 |

## FIR conclusion

- Bypassing the existing FIR causes severe NMSE and ACLR degradation, so the
  FIR remains essential for delay and linear-response equalization.
- Frequency-weighted closed-form FIR refits with adjacent weights from 1 to
  4096 leave the 2284 full-band ACLR near -42.995 dBc while NMSE degrades.
- FIR43/FIR55/FIR95 checkpoints have similar new-metric ACLR. Increasing FIR
  length or adjacent fitting weight is therefore not the route to -45 dBc.

## Architecture conclusion

The H32 model proves that all three metrics can beat the dataset reference under
an NMSE <= -40 dB constraint. Existing 2284/998/786/796 architectures cannot
transfer that spectrum through ordinary fine-tuning or prediction distillation.
The next experiment should train from scratch with the Welch-consistent
constraint and add a small subband residual cancellation branch, while retaining
the existing FIR for equalization.
