# Constrained multi-resolution final selection (2026-07-16)

Each model directory contains the selected `best.pth`, reconstruction sidecars,
complete candidate search/history, raw+smoothed PSD comparison, NMSE convergence
plot, and an artifact manifest. The deployable architecture and multiplier count
are unchanged from the corresponding source wrapper.

The numbers are validation-selection scores on the existing 0.8 split, not an
independent test claim. `786` is a historical model label; its audited deployment
cost is 798 real multipliers, which remains within the 800 limit.

See `champions.json` for the exact selected source, MAC audit, and 2048/4096/8192
metrics. Full positive and negative trials remain in sibling directories:
`multires_integrated_r1`, `multires_band_r1`, `multires_band_strict`,
`multires_band_qualified`, `continuous_range_r1`, and
`continuous_range_qualified`.
