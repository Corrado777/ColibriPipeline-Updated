# A plain-language guide to the occultation-detection trade study

This explains, from the ground up, what the `detection_trade_study/` harness does,
every term used in the results, how to read each figure, what we concluded, and
what to actually change in the production pipeline. No prior signal-processing
background assumed.

---

## 1. What are we trying to detect?

We are looking for **stellar occultations by sub-km trans-Neptunian objects
(TNOs)**. When a tiny, distant object passes in front of a background star, the
star's brightness briefly drops. Because the objects are small and far away, the
dip is **sub-second** and is shaped by **Fresnel diffraction** (it has a
characteristic dip-with-ringing shape, not a clean box). At 40 Hz (one frame
every 0.025 s) such an event lasts only a handful of frames.

A "light curve" is just the brightness of one star measured every frame for a
minute (~2399 numbers). We have ~1287 stars per telescope per minute, across
three telescopes (Red / Green / Blue) that watch the same sky at the same time.

The production pipeline today:
1. Builds each star's light curve (`colibri_photometry.py`).
2. **Detrends** it (subtracts a 5-second running-mean **boxcar**) and then
   convolves the residual with a single fixed-width **Ricker wavelet** matched
   filter, flagging dips above a threshold (`dipDetection`). Note: the detrend
   uses a "boxcar" (running mean) purely as a high-pass filter; the detection step
   uses a separate template (Ricker wavelet or, in this study, also box-shaped
   templates). These are two distinct uses of the word "boxcar" — see §3.
3. After each telescope independently flags dips, it requires the same event to
   appear in 2-3 telescopes at the same time/place (`simultaneous_occults.py`).

The trade study asks three questions:
- Is the fixed-width Ricker the best detector, or is something else better?
- Are we using the three telescopes as powerfully as we could?
- Can better **detrending** improve sensitivity?

---

## 2. The core idea: injection and recovery

We cannot test a detector on real occultations because we have almost none. So we
do **injection/recovery**:

1. Take a **real** star's light curve (real noise, real systematics).
2. **Inject** a synthetic Fresnel dip of known depth and timing into it
   (`injection.py`, using the real diffraction physics in `fresnel_physics.py`).
3. Run a detector and ask: did it find the dip, at the right time?
4. Repeat thousands of times over many stars, depths, and timings.

The fraction of injected events we recover is the detector's **completeness**
(a.k.a. detection efficiency). High completeness = sensitive detector.

But completeness alone is meaningless without controlling **false alarms** — a
detector that screams "dip!" on every frame would have 100% completeness and be
useless. So we also run **null trials** (no injection) to measure how often the
detector fires on pure noise.

---

## 3. Glossary (the terms in the plots)

**Detrending / high-pass.** A light curve slowly wanders (the star rises/sets,
clouds, tracking). We remove that slow wander so a short dip stands out. The
current method ("MeanSubtract") computes a 5-second running average and subtracts
it — what's left is the fast wiggles, including any occultation. "High-pass"
because it keeps fast (high-frequency) changes and removes slow ones.

**Two uses of "boxcar" — don't confuse them.** The word "boxcar" appears in two
completely different contexts here: (1) the *detrend boxcar* is the 5-second
running mean that is *subtracted* from the light curve to flatten slow trends —
it is a preprocessing step, not a detector; (2) the *box detection statistic*
(implemented as `BoxDetector` / `MultiWidthBoxDetector`) sums the flux over a
short N-frame window *after* detrending and looks for a depression relative to
the local mean — this is a geometric dip template used as the matched-filter
shape. A `BoxDetector(6)` looks for a flat, 6-frame (0.15 s) dip; it has nothing
to do with the 5-second detrend window.

**Matched filter.** The optimal way to find a known shape buried in noise: slide a
template of that shape along the data and measure how well it matches at each
point. A Ricker wavelet, a box, or a Fresnel template are all just template
shapes. The output is a "score" at every frame.

**Significance / sigma (σ).** The detector's score at a frame, expressed in units
of the local noise. "5σ" means the dip is 5 times larger than the typical noise
fluctuation. Higher = more convincing.

**Threshold.** We declare a detection when the score exceeds some cutoff. A high
threshold = few false alarms but you miss weak events; a low threshold = catch
weak events but lots of false alarms. This is the fundamental trade-off.

**False-Alarm Rate (FAR).** How often the detector fires when there is *no* real
event. Here we measure it on the null trials as a fraction: **"1% FAR" means we
set the threshold so the detector falsely fires on only 1% of star-minutes with
no injection.** It is the leash on the detector. We always compare detectors *at
the same FAR* — otherwise it's not a fair fight (a trigger-happy detector looks
"more complete" only because it fires constantly).

**Completeness (at a given FAR).** With the threshold set to give, say, 1% FAR,
what fraction of *injected* events do we recover? This is the number we care
about: real sensitivity at a controlled false-alarm budget.

**ROC curve (Receiver Operating Characteristic).** Sweep the threshold from strict
to loose and plot **completeness (y) vs FAR (x)**. Each point is one threshold
setting. A better detector sits higher and to the left (more completeness for less
false-alarm). "Completeness at 1% FAR" is just reading one vertical slice of the
ROC at x = 0.01. The name is historical (from WWII radar); think of it as the
"sensitivity-vs-false-alarm trade-off curve."

**Event SNR (signal-to-noise).** A single number summarizing how detectable an
injected event *should* be: roughly `depth × stellar_SNR × √(event_length)`. A
deep dip on a bright star lasting several frames has high event SNR; a shallow dip
on a faint star has low event SNR. We expect completeness to rise sharply once
event SNR crosses a threshold — and it does (around event SNR ≈ 17-27 here).

**Stellar (per-frame) SNR.** How bright a star is relative to its frame-to-frame
noise: `median_flux / noise`. Bright stars have high SNR (here up to ~10); faint
stars ~1. This turned out to be the dominant factor.

---

## 4. Why some completeness-map boxes are white (blank)

The completeness map is a grid: rows = stellar SNR bins, columns = event-depth
bins, color = completeness in that cell. **A white/blank cell means that cell had
no injected trials to average** (completeness is undefined → NaN → drawn blank).

Two reasons a cell can be empty:
- **No injections landed there.** Injected depth comes from sampling physical
  Fresnel parameters; very deep events (depth > 0.8) only occur for certain
  object sizes/impact parameters, so those columns are sparsely populated. And
  high-SNR stars are rare in the pool, so the top rows have few trials.
- **Too few trials overall.** With a few hundred trials spread over a 6×5 grid,
  some cells get 0-2 injections by chance.

It is *not* a bug — it just means "we didn't sample this combination enough to
estimate completeness." The fix is more trials (`--n-grid`) or coarser bins. Each
cell's trial count is in the `counts` array returned by `harness.completeness_map`.

---

## 5. The modules, in order of the data flow

```
fresnel_physics.py   real diffraction-curve generator (the truth signal shape)
injection.py         sample event params -> transmission profile -> multiply into real flux
lightcurves.py       load real stars.npy; clean; star_snr; cross-match scopes
preprocessing.py     condition the curve: MeanSubtract / MedianDivide / HighPass / Bandpass / Whiten
matched_filter.py    slide a template, return a per-frame significance (handles whitening)
detectors.py         a detector = (preprocessing) x (template shape); 6 shapes; build_grid()
bootstrap.py         make N independent fake telescopes from one real star (for the 3-scope test)
combine.py           joint statistic (sum scores / sqrt(N)) and the AND reference
harness.py           run thousands of inject/null trials; compute ROC, completeness maps
run_trade_study.py   CLI: run everything, write CSVs + figures + tables
```

Two pieces are swappable so we can trade-study them:
- **Preprocessing** (how we flatten the curve and model its noise).
- **Template shape** (what dip shape we match: Ricker, box, multi-width Ricker,
  multi-width box, Fresnel bank, single-frame).

`build_grid()` runs every preprocessing × shape combination.

---

## 6. The two experiments and how to read each figure

### Experiment A — single-telescope detector & preprocessing comparison
- `roc_single.png` — ROC for each detector shape. Higher/left = better.
- `preprocessing_heatmap.png` — completeness at 1% FAR for every
  preprocessing × shape combo. Brighter = better.
- `completeness_vs_eventSNR.png` — completeness vs event SNR, one line per
  preprocessing method. Shows the detection threshold (the S-curve rise).
- `completeness_map.png` — the 2D map of completeness over (stellar SNR × depth).
  **This is the most important figure**: it shows completeness is high for deep
  events on bright stars and ~0 for faint stars, which a single pooled number
  hides.

### Experiment B — "power of three"
- `power_of_three.png` — for each detector, three ROC curves:
  - **single** = one telescope.
  - **AND** = the current scheme: each telescope thresholds independently, then we
    require a coincident detection.
  - **joint** = sum the per-frame scores of all telescopes *before* thresholding
    (divided by √N so noise stays calibrated), then threshold once.
  Because the test minute's three scopes are actually identical copies (a
  data-quality issue, see §8), independent telescopes are *simulated* by
  bootstrapping one real star's noise (`bootstrap.py`).

---

## 7. What we found

1. **Completeness at large depth is NOT bad — it was a reporting artifact.** The
   SNR-binned map shows ~100% recovery of deep events on stars with per-frame
   SNR ≳ 6, dropping to ~0 on faint stars. The single pooled "~7%" number was
   averaging the excellent bright-star corner with the hopeless faint-star bulk.

2. **The bottleneck is stellar SNR, not detrending.** The noise here is dominated
   by high-frequency, per-frame scatter (the 5-second detrend changes the noise by
   <2% — there is almost nothing slow to remove). You cannot detrend away
   per-frame photon noise. Median stellar SNR is ~4.5 and **no star exceeds 10**.

3. **Detector shape barely matters on this data.** Box ≈ Fresnel ≥ single Ricker;
   the multi-width Ricker bank did *not* beat a single width; the single-frame
   "normalized SNR" was worst. Differences are a few percent.

   > **Important caveat — the Ricker "width 6" is not 0.15 s.**
   > `astropy.convolution.RickerWavelet1DKernel(width)` takes a *scale parameter*,
   > not a number of frames. `RickerWavelet1DKernel(6)` produces a **49-sample
   > (~1.23 s) bipolar wavelet** at 40 Hz. The production pipeline builds this
   > kernel intending `expected_length=0.15` s (`kernel_frames=6`) but actually
   > uses a kernel ~8× wider — likely **too broad for the shortest sub-second
   > events**. The box templates ARE specified directly in frames, so `BoxDetector`
   > and `MultiWidthBoxDetector` widths are exact frame counts. This means a
   > box-vs-Ricker comparison at "width 6" was *not* comparing equal durations;
   > the box at 0.15 s was being tested against a Ricker at ~1.2 s. The
   > competitive (or better) box performance should be interpreted in that light.

4. **Preprocessing barely matters here either** — all five methods track within a
   few percent, because the noise is white-floor-limited. **Whitening was the best
   at the high-event-SNR end** (0.73 vs 0.64 for Ricker), consistent with the mild
   colored-noise we measured (lag-1 autocorrelation 0.30). On data with stronger
   colored noise, whitening's edge would grow.

5. **The three-telescope joint statistic is the real win.** Combining per-frame
   scores before thresholding roughly **doubled-to-tripled** completeness over a
   single telescope at fixed FAR, and clearly beat the current post-threshold AND
   coincidence. A signal that is ~3σ in each scope becomes ~5σ jointly, while
   independent false alarms do not stack.

---

## 8. Caveats about the test data (why some design choices look odd)

The bundled minute (`2025-08-30 01.54.12`) is degraded, which shaped the harness:
- **Green and Blue are byte-identical copies** → can't test multi-telescope on
  them directly → we bootstrap independent noise instead.
- **Red is a 5-second partial** (199 frames) → excluded.
- **Timestamps are quantized to whole seconds** → telescopes are aligned by frame
  index, not time interpolation.
- **~30% of "stars" have negative median flux** → only positive-flux stars
  (median ≥ 500 counts) are used as injection hosts.

So treat the *absolute* completeness numbers as illustrative; the *relative*
comparisons (shape vs shape, joint vs AND, whitening at high SNR) are the result.
Re-running on a clean minute with three genuinely independent telescopes is the
natural next validation.

---

## 9. Conclusions and what to change in the production pipeline

**Your direct questions:**

> Should we just do a boxcar?

Yes — keep the boxcar detrend (`MeanSubtract`). It is not the bottleneck and is as
good as anything else on this data. Don't spend effort replacing it. (If you later
find strongly colored noise on real data, revisit whitening — it had the edge at
high SNR — but that's a refinement, not a priority.)

> Do we do multiple widths?

For **Ricker** widths: the multi-width bank did **not** beat a single width in
these trials. For **box** widths: this is now being evaluated with the new
`MultiWidthBoxDetector` — a BLS-style bank of flat-dip templates at widths 2, 3,
4, 6, 9, 14, 20, 32 frames (0.05–0.8 s at 40 Hz), taking the per-frame maximum
significance over all widths. Whether scanning multiple box widths beats a single
appropriately-sized box is an open question; see the regenerated
`preprocessing_heatmap.png` and `completeness_vs_eventSNR.png` in `results/`
rather than relying on any number quoted here. The single-width `BoxDetector` is
already a simpler, more interpretable baseline that tested competitively — if
`MultiWidthBoxDetector` offers no meaningful gain over it, the single-width box
is the right choice.

If you change the matched-filter shape at all, a **Fresnel template bank** (which
you already have in `colibri_secondary.py`) is the principled choice and was tied
for best — but the single-telescope gain is small.

**Production-pipeline note on Ricker kernel width.** Before choosing any
matched-filter shape for production, fix the kernel-sizing bug: the production
`dipDetection` builds `RickerWavelet1DKernel(6)` intending a 0.15 s kernel but
actually gets a ~1.23 s, 49-sample wavelet. For short sub-second events the
production Ricker is almost certainly over-broad. Either (a) correct the
`RickerWavelet1DKernel` argument (the `width` parameter needs to be ~1, not 6,
to approximate a 6-frame FWHM), or (b) switch to the box template, which IS
specified directly in frames and is correctly sized.

> What is the main conclusion for my production pipeline?

The two highest-value changes are **not** about detrending or wavelet shape:

1. **Use the three telescopes jointly, not post-threshold.** Today each telescope
   thresholds at high σ and then `simultaneous_occults.py` requires a coincidence.
   Instead, save each telescope's *per-frame detection score* (or normalized
   flux), align the three by GPS time, sum them (÷√N), and threshold the joint
   series once. This lets each telescope run at a *lower* per-scope threshold at
   the same joint false-alarm rate — the biggest sensitivity gain we measured.

2. **Sensitivity is set by stellar brightness.** You are photon-floor-limited.
   Focus on the brighter stars for the sub-km search (or improve the photometry
   SNR — e.g. the aperture flux currently has no local background subtraction,
   which adds sky-noise; that's an upstream change worth quantifying separately).

### How to implement (concrete)

- **Keep `dipDetection` essentially as-is** (boxcar detrend + a matched filter).
  Optionally swap the single Ricker for a Fresnel matched-filter bank reusing the
  kernels in `colibri_secondary.py`; expect only a small single-scope gain.
- **Joint statistic (the important one):**
  - In `colibri_main_py3` / `colibri_photometry`, additionally **persist the
    per-frame detection score** (the normalized convolution output) per star, not
    just the final flagged events.
  - In `simultaneous_occults.py`, replace the post-threshold time/coordinate
    AND-match with: for each star matched across telescopes, align the per-frame
    scores on a common GPS-time grid, form `joint = Σ score_i / √N`, and flag
    where the joint series exceeds a threshold tuned to the desired joint FAR. The
    `combine.py` module in this harness is a working reference implementation
    (`joint_score`, `joint_peak`).
  - Re-tune the per-telescope and joint thresholds using this harness's ROC on a
    clean, real, independent-telescope minute before deploying.

In short: **the wavelet/detrending debate is a side-show on this data; keep the
boxcar and a simple matched filter, and put the engineering effort into a joint
three-telescope statistic and brighter-star selection.**

---

## 10. Running and extending

```bash
cd ColibriPipeline
python -m detection_trade_study.run_trade_study --n-inject 300 --n-null 300 --n-grid 250
# writes CSVs + PNGs to detection_trade_study/results/
```
Interactive walk-through: `notebooks/detection_trade_study.ipynb`.

- **Add a detector shape:** add a dip-template factory (dip = negative values) in
  `detectors.py` and register it in `_shape_templates()`.
- **Add a preprocessing method:** subclass `preprocessing.Preprocessor`, return a
  `Conditioned(series, noise)`, and add it to `ALL_PREPROCESSORS()`.
- **Test true three-telescope data:** call `harness.run_trials_real` with three
  genuinely independent minutes instead of the bootstrap path.

See `README.md` for the module-by-module reference.
