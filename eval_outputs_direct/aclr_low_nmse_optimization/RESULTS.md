# Low-NMSE-impact ACLR optimization results

All experiments keep the deployed model architecture unchanged.  The residual is
`error = target_y - pred_y`.  Evaluation uses Welch PSD with the following three
diagnostics:

- full-band mean: target main 20--80 MHz, error adjacent -180---120/120--180 MHz;
- minimum point in the same three bands;
- fixed 5 MHz narrow bands: 31.64--36.64, -129.84---124.84 and 121.08--126.08 MHz.

The dataset reference is `-43.7378/-45.1045/-44.9965 dBc` for
full/minimum/narrow.

## Methods tested on 2284

| Method | NMSE dB | Full dBc | Minimum dBc | Narrow dBc | Observation |
|---|---:|---:|---:|---:|---|
| Release baseline | -40.152 | -42.980 | -44.717 | -43.834 | Starting point |
| Trust-region, ratio 1 | -40.171 | -43.065 | -44.920 | -43.945 | Stable small gain |
| PCGrad, ratio 1 | -40.175 | -43.061 | -44.906 | -43.936 | Similar because most gradient cosines were positive |
| Backbone/FIR alternating | -40.172 | -43.037 | -44.808 | -43.926 | Weaker than joint updates |
| Strong PCGrad refinement | -40.098 | **-43.213** | **-45.108** | **-44.085** | Best stable full-band result |

EMA and SWA were evaluated for every strategy.  On the strong 2284 run, raw/EMA/SWA
full-band ACLR was `-43.213/-43.202/-43.185 dBc`; averaging slightly improved NMSE
but did not improve ACLR.  A 0.05-step interpolation scan also selected alpha 1.0,
so no additional interpolation gain existed beyond the selected trained state.

## Final four-model comparison

| Model | Baseline NMSE | Final NMSE | Baseline full/min/narrow | Final full/min/narrow | Full gain | Narrow gain |
|---|---:|---:|---|---|---:|---:|
| 2284 | -40.152 | -40.098 | -42.980/-44.717/-43.834 | **-43.213/-45.108/-44.085** | 0.233 dB | 0.251 dB |
| 998 | -37.366 | -37.262 | -40.089/-41.975/-41.383 | **-40.317/-42.171/-41.669** | 0.228 dB | 0.286 dB |
| 786 | -35.462 | -35.406 | -38.512/-40.272/-39.146 | **-38.697/-40.745/-39.386** | 0.185 dB | 0.240 dB |
| 796 | -35.199 | -35.200 | -38.026/-40.182/-38.689 | **-38.207/-40.595/-38.927** | 0.181 dB | 0.238 dB |

All final candidates satisfy their NMSE constraints (-40/-37/-35/-35 dB).  The
NMSE change relative to the release baselines is only +0.054/+0.104/+0.056/-0.001
dB for 2284/998/786/796 respectively.

## Conclusion

The most effective non-architectural method is joint fine-tuning with an explicit
NMSE trust gate, a parameter-distance trust region, and projected spectral
gradients.  FIR-only recovery, EMA, SWA and post-training interpolation are useful
safety checks but did not create the main ACLR gain.

For 2284, the minimum-point metric now narrowly beats the dataset reference by
0.003 dB.  The full and narrow metrics remain 0.525 and 0.912 dB above the dataset
reference, respectively.  The lighter models remain several dB from the dataset
reference; the experiments therefore do not support a claim that optimization
alone can drive those architectures to -45 dBc while retaining their NMSE.

## Main artifacts

- `steps/aclr_low_nmse_optimization.py`: reproducible experiment runner.
- `refine_2284_pcgrad20/2284/pcgrad/best_interpolated.pth`: selected 2284 checkpoint.
- `transfer_light_pcgrad20/<model>/pcgrad/best_interpolated.pth`: selected light checkpoints.
- `final_unified_eval/<model>/`: independent unified summaries and PSD plots.
