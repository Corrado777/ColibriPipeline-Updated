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
| `lightcurves.py` | Load/clean `stars.npy`; cross-match stars; `star_snr` (per-frame SNR). |
| `preprocessing.py` | Swappable conditioning: `MeanSubtract` (baseline), `MedianDivide`, `HighPass`, `Bandpass`, `Whiten` (PSD). Each returns a flat residual + a noise model. |
| `matched_filter.py` | One optimal-filter core handling scalar / per-frame / PSD (whitened) noise models. |
| `detectors.py` | `DetectionResult` contract + `MatchedFilterDetector` (preprocessing × dip-template shape) + `ALL_DETECTORS()` + `build_grid()`. |
| `bootstrap.py` | Synthesise N independent telescopes from one real star by moving-block bootstrap of its residual noise. |
| `combine.py` | Align per-scope scores, joint statistic, AND reference. |
| `harness.py` | Injection/recovery loops + ROC + SNR-aware `completeness_map` / `completeness_vs_eventSNR`. |
| `run_trade_study.py` | CLI driver → CSVs + PNG figures + summary tables. |

Detection is factored into a **preprocessing** step × a **dip-template shape**:
- Shapes: `RickerDetector` (production baseline), `BoxDetector` (geometric),
  `MultiWidthRickerDetector`, `FresnelMatchedFilterDetector` (40 Hz Fresnel bank),
  `NormalizedSNRDetector` (single-frame / Pass-style).
- Preprocessors: `MeanSubtract`, `MedianDivide`, `HighPass`, `Bandpass`, `Whiten`.
- `ALL_DETECTORS()` = the five shapes on the default `MeanSubtract`.
  `build_grid()` = the full {shape}×{preprocessing} grid (names `shape@prep`).

## Adding a method

* New **shape**: add a dip-template factory (dip = negative) in `detectors.py`
  and an entry in `_shape_templates()`.
* New **preprocessing**: subclass `preprocessing.Preprocessor`, return a
  `Conditioned(series, noise)` (noise = `('scalar',σ)`, `('perframe',σ[])`, or
  `('psd',P,fs)`), and add it to `ALL_PREPROCESSORS()`.
Both then appear automatically in `build_grid()` and every ROC/joint policy.

## Why completeness looked "miserable" — and the SNR-aware view

The pooled completeness was misleadingly low because the noise here is mostly
**high-frequency / per-frame** (the boxcar detrend changes the std by <2%), and
the host pool is dominated by **faint** stars (median per-frame SNR ≈ 4.5; none
exceeds 10). Detrending can't remove white noise, and a deep dip on a faint star
sits at the noise floor. The detectors are fine — on bright hosts (SNR 7–10) with
deep events, localization is ~100%. The noise *is* mildly colored (lag-1
autocorrelation ≈ 0.30), which is why `Whiten`/`Bandpass` can help.

So the report is now **SNR-aware**: `completeness_map` (stellar SNR × event depth)
and `completeness_vs_eventSNR` (vs `depth·SNR·√duration`) instead of one pooled
number — see `completeness_map.png`, `completeness_vs_eventSNR.png`,
`preprocessing_heatmap.png`.

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
