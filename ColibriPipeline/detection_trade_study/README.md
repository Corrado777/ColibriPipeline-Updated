# Occultation-detection trade study (offline)

A standalone, offline harness for comparing occultation-detection algorithms and
quantifying the multi-telescope ("power of three") sensitivity gain. **Nothing
here is imported by the live pipeline** — it reads `stars.npy` light curves and
runs experiments; it never writes to `ColibriData`/`ColibriArchive` or changes a
science stage.

## What it answers

1. **Is the production Ricker the best primary detector?** Injects synthetic
   Fresnel signals into real light curves and compares five detectors on the same
   ROC (completeness vs false-alarm rate).
2. **Are we leaving sensitivity on the table by combining telescopes only after
   thresholding?** Compares single-telescope, post-threshold **AND** (the current
   `simultaneous_occults` scheme), and a **joint statistic** (sum of per-frame
   scores / √N, thresholded once).

## Modules

| file | role |
|------|------|
| `fresnel_physics.py` | Qt-free copy of the Lommel-diffraction generator (`genCurve`) from `KernelGeneratorGUI_RAB032922/fresnelModelerGUI.py`. The original imports PySide2 and can't be imported headless. |
| `injection.py` | Sample Fresnel params, build a multiplicative transmission profile, inject into a real flux series. |
| `lightcurves.py` | Load/clean `stars.npy`; cross-match stars across scopes. |
| `detectors.py` | `DetectionResult`/`Detector` contract + 5 detectors + `ALL_DETECTORS()`. |
| `bootstrap.py` | Synthesise N independent telescopes from one real star by moving-block bootstrap of its residual noise. |
| `combine.py` | Align per-scope scores, joint statistic, AND reference. |
| `harness.py` | Injection/recovery trial loops (`run_trials_real`, `run_trials_bootstrap`) + ROC. |
| `run_trade_study.py` | CLI driver → CSVs + PNG figures + summary tables. |

The five detectors: `RickerDetector` (production baseline), `BoxDetector`
(geometric), `MultiWidthRickerDetector`, `FresnelMatchedFilterDetector` (uses the
40 Hz Fresnel kernel bank), `NormalizedSNRDetector` (Pass-style per-frame z-score).

## Adding a detector

Subclass `detectors.Detector`, set `name`, implement
`run(flux, exposure_time) -> DetectionResult` returning a **full-length per-frame
`score`** (NaN-padded edges allowed), and add it to `ALL_DETECTORS()`. It is then
automatically included in both experiments and all ROC/joint policies.

## Running

```bash
cd ColibriPipeline
python -m detection_trade_study.run_trade_study --n-inject 300 --n-null 300
# figures + CSVs land in detection_trade_study/results/
```

`notebooks/detection_trade_study.ipynb` reproduces the figures interactively.

## Important caveats about the bundled test minute (2025-08-30 01.54.12)

These shaped the design and matter for interpreting results:

- **Green and Blue are byte-identical copies** (correlated noise). A joint
  statistic over them is meaningless, so the power-of-three experiment uses
  **block-bootstrapped independent noise** drawn from one real star's residuals
  (same real noise statistics, independent realizations, coincident injected
  signal). The harness still supports a true `run_trials_real` multi-scope join
  for when genuine independent per-telescope minutes are available.
- **Red is a 5 s partial capture** (199 frames) — shorter than the 200-frame
  detrend window — so it is excluded from the combination study.
- **GPS timestamps are quantised to whole seconds** (61 unique values for 2399
  frames), so joint alignment is done by **frame index**, not time interpolation.
- **~30 % of stars have negative median flux** (background-subtracted sim
  artifacts); only positive-flux stars (`photometric_hosts`, median ≥ 500 counts)
  are used as injection hosts.
