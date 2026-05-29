"""
CLI driver for the occultation-detection trade study.

Runs the full injection/recovery experiment on the bundled test minute and
writes results + figures:

  results.csv                      -- the raw trial table (one row per trial)
  roc_single.png                   -- single-telescope ROC, all detectors
  completeness_vs_depth.png        -- single-telescope completeness vs depth
  power_of_three.png               -- single vs joint vs AND for each detector

Usage:
    python -m detection_trade_study.run_trade_study \
        [--n-inject 300] [--n-null 300] [--seed 0] [--outdir <dir>]

It imports only stdlib + numpy + scipy + astropy + pandas + matplotlib, all of
which are already available.  Nothing here touches the live pipeline.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import harness
from . import detectors as det_mod


def _plot_roc_single(roc, outpath):
    """Single-telescope ROC (completeness vs false-alarm) for every detector."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for det, policies in roc.items():
        e = policies['single']
        order = np.argsort(e['far'])
        ax.plot(e['far'][order], e['completeness'][order], marker='.', label=det)
    ax.set_xscale('symlog', linthresh=1e-2)
    ax.set_xlabel('False-alarm rate (per star-minute)')
    ax.set_ylabel('Completeness (recovery fraction)')
    ax.set_title('Single-telescope detector trade study')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def _plot_completeness_vs_depth(df, detectors, ref_scope, outpath, n_bins=8):
    """Completeness vs injected depth at a fixed per-detector threshold.

    The threshold per detector is its 99th-percentile null peak (i.e. ~1% FAR),
    so the curves are comparable at matched false-alarm rate.
    """
    inj_df = df[df['kind'] == 'inject']
    null_df = df[df['kind'] == 'background']
    depth = inj_df['depth'].values
    bins = np.linspace(depth.min(), depth.max(), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(7, 6))
    for det in detectors:
        peak_col = f'{det}|{ref_scope}|peak'
        frame_col = f'{det}|{ref_scope}|frame'
        T = np.nanpercentile(null_df[peak_col].values, 99)
        rec = (inj_df[peak_col].values >= T) & (
            np.abs(inj_df[frame_col].values - inj_df['center_frame'].values) <= 12)
        comp = []
        for i in range(n_bins):
            m = (depth >= bins[i]) & (depth < bins[i + 1])
            comp.append(rec[m].mean() if m.any() else np.nan)
        ax.plot(centers, comp, marker='o', label=det)
    ax.set_xlabel('Injected event depth (1 - min transmission)')
    ax.set_ylabel('Completeness at ~1% FAR')
    ax.set_title('Completeness vs injected depth (single telescope)')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def _plot_power_of_three(roc, outpath):
    """For each detector, overlay single / joint / AND ROC curves."""
    dets = list(roc.keys())
    ncol = min(3, len(dets))
    nrow = int(np.ceil(len(dets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow),
                             squeeze=False)
    styles = {'single': ('1 telescope', 'tab:blue'),
              'and': ('3 scope: post-threshold AND', 'tab:orange'),
              'joint': ('3 scope: joint statistic', 'tab:green')}
    for k, det in enumerate(dets):
        ax = axes[k // ncol][k % ncol]
        for pol, (label, color) in styles.items():
            e = roc[det][pol]
            order = np.argsort(e['far'])
            ax.plot(e['far'][order], e['completeness'][order],
                    marker='.', color=color, label=label)
        ax.set_xscale('symlog', linthresh=1e-2)
        ax.set_title(det, fontsize=10)
        ax.set_xlabel('False-alarm rate')
        ax.set_ylabel('Completeness')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    # blank any unused axes
    for k in range(len(dets), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle('Power of three: joint statistic vs post-threshold AND',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-inject', type=int, default=300)
    ap.add_argument('--n-null', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--source-scope', default='Green',
                    help='Full-minute scope used as the real noise source.')
    ap.add_argument('--n-scopes', type=int, default=3,
                    help='Bootstrapped telescopes for the power-of-three study.')
    ap.add_argument('--min-median-flux', type=float, default=500.0)
    ap.add_argument('--outdir', default=str(Path(__file__).parent / 'results'))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    detectors = det_mod.ALL_DETECTORS()
    rng = np.random.default_rng(args.seed)

    # --- Data inventory + quality report --------------------------------
    print("Loading telescopes...")
    data_by_scope, _, all_scopes = harness.load_all()
    frame_counts = {s: data_by_scope[s]['flux'].shape[1] for s in all_scopes}
    full = harness.select_full_scopes(data_by_scope, all_scopes)
    partial = [s for s in all_scopes if s not in full]
    identical = harness.scopes_identical(data_by_scope, full)
    print(f"  scopes={all_scopes}  frame counts={frame_counts}")
    if partial:
        print(f"  partial captures (excluded): { {s: frame_counts[s] for s in partial} }")
    if identical:
        print(f"  WARNING: full scopes {full} carry IDENTICAL flux (correlated "
              f"noise) -> a raw multi-scope join is not meaningful; the power-of-"
              f"three study uses block-bootstrapped independent noise instead.")

    if args.source_scope not in full:
        args.source_scope = full[0]
    source = data_by_scope[args.source_scope]
    hosts = harness.photometric_hosts(source['flux'], args.min_median_flux)
    print(f"  noise source = {args.source_scope}; "
          f"{len(hosts)} positive-flux host stars (median >= {args.min_median_flux:g}).")

    # One bootstrap experiment provides BOTH the single-telescope detector
    # comparison (its 'single' policy) and the power-of-N comparison.  The
    # bootstrap substrate gives a stationary real-noise background, which is the
    # right basis for a controlled detector trade study; injecting into the raw
    # non-stationary curves instead just measures how badly real systematics
    # (e.g. the large shared spikes in this minute) swamp single-telescope
    # thresholding -- a separate, known problem that coincidence is meant to fix.
    print(f"\nRunning {args.n_scopes} bootstrapped telescopes x "
          f"{args.n_inject} inject + {args.n_null} null trials "
          f"x {len(detectors)} detectors...")
    df, scopes, ref = harness.run_trials_bootstrap(
        source, hosts, detectors, n_scopes=args.n_scopes,
        n_inject=args.n_inject, n_null=args.n_null, rng=rng)
    df.to_csv(outdir / 'results_multi.csv', index=False)
    roc = harness.compute_roc(df, scopes, ref, detectors=list(detectors.keys()))

    _plot_roc_single(roc, outdir / 'roc_single.png')
    _plot_completeness_vs_depth(df, list(detectors.keys()), ref,
                                outdir / 'completeness_vs_depth.png')
    _plot_power_of_three(roc, outdir / 'power_of_three.png')
    print(f"  wrote results + figures to {outdir}")

    # --- Headline summary tables at ~1% FAR -----------------------------
    print("\n== Single-telescope detector comparison: completeness at ~1% FAR ==")
    print(f"{'detector':30s} {'completeness':>12s}")
    for det in detectors:
        c = harness.completeness_at_far(roc[det]['single'], 0.01)
        print(f"{det:30s} {c:12.2f}")

    print(f"\n== Power of {args.n_scopes}: completeness at ~1% false-alarm rate ==")
    print(f"{'detector':30s} {'1-scope':>8s} {'AND':>8s} {'joint':>8s}")
    for det in detectors:
        s = harness.completeness_at_far(roc[det]['single'], 0.01)
        a = harness.completeness_at_far(roc[det]['and'], 0.01)
        j = harness.completeness_at_far(roc[det]['joint'], 0.01)
        print(f"{det:30s} {s:8.2f} {a:8.2f} {j:8.2f}")


if __name__ == '__main__':
    main()
