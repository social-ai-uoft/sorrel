#!/usr/bin/env python3
"""Study1 vs Study2 phase-progression contrast (D / N_rot).

NOTE — defaults:
  s = Global_average_punishment_level
  x = Mean_mean_a_encounters / sum_r Mean_mean_r_encounters
      (resource A share of total resources extracted; NOT raw A)
  detrend = linear (after windowing; use --detrend none to keep secular drift)
  start_after = 40000 (keep t >= 40000; use --start-after 0 for full series)
  last_n = 0 (no tail truncation; optional --last-n N)
  smooth_w = 10 (use --smooth-w 1 for unsmoothed; --smooth-w 51 for older default)
  contrast figure: unwrapped θ | raw D | D_smooth | N_rot_smooth
  Override with --x-mode raw_a if absolute A intensity is needed.

Computes angular-consistency metrics, writes per-run diagnostics, a metrics
CSV, and a Study1-vs-Study2 contrast figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_angular_consistency_nrot_standalone import (
    phase_progression_metrics,
    plot_phase_diagnostics,
)
from system_analysis_utils import list_run_dirs, load_s_x_series, resolve_explicit_runs

STUDY1 = "Study1_visible_s"
STUDY2 = "Study2_hidden_s"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-progression metrics + Study1 vs Study2 contrast plots."
    )
    p.add_argument(
        "--analysis-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory that contains the res/ folder (default: this script's dir).",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Explicit run directories or basenames (overrides --runs-dir discovery).",
    )
    p.add_argument(
        "--runs-dir",
        type=str,
        default="res/runs_bandit_study2_v3_twoArms",
        help="Relative (to analysis-root) or absolute path to run folders.",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=["copy"],
        help="Exclude run names containing any of these substrings.",
    )
    p.add_argument(
        "--run-indices",
        nargs="*",
        type=int,
        default=None,
        help="Which discovered runs to use (ignored when --runs is set).",
    )
    p.add_argument(
        "--condition-tags",
        nargs="*",
        default=None,
        help=(
            "Optional per-run labels. Also used as study-group hints when a tag "
            "contains 'study1'/'pknown' or 'study2'/'punkn'."
        ),
    )
    p.add_argument(
        "--study-groups",
        nargs="*",
        default=None,
        help=(
            f"Explicit group labels per run (must match run count). "
            f"Preferred values: {STUDY1} / {STUDY2}."
        ),
    )
    p.add_argument(
        "--start-after",
        type=int,
        default=40_000,
        help=(
            "Drop early burn-in: keep only steps with t >= N "
            "(default 40000 = all steps after 40000). Set 0 for full series."
        ),
    )
    p.add_argument(
        "--last-n",
        type=int,
        default=0,
        help=(
            "Optional tail window: use only the last N steps after --start-after "
            "(default 0 = no tail truncation)."
        ),
    )
    p.add_argument(
        "--x-mode",
        type=str,
        default="a_share",
        choices=["a_share", "raw_a"],
        help=(
            "Phase x observable. Default a_share = A/total resource encounters "
            "(preferred). raw_a = absolute Mean_mean_a_encounters."
        ),
    )
    p.add_argument(
        "--detrend",
        type=str,
        default="linear",
        choices=["none", "linear", "rolling"],
        help=(
            "Detrend s and x after start-after/last-N windowing, before "
            "smoothing/z-score. Default linear removes least-squares drift "
            "(orbit-focused); use none to keep secular change in N_rot/D."
        ),
    )
    p.add_argument(
        "--detrend-rolling-w",
        type=int,
        default=None,
        help="Rolling-mean width when --detrend rolling (default ~n/10).",
    )
    p.add_argument(
        "--smooth-w",
        type=int,
        default=10,
        help="Rolling smooth width on s/x before angle (default 10; 1 = unsmoothed).",
    )
    p.add_argument("--n-surr", type=int, default=0, help="Surrogate nulls (0=off).")
    p.add_argument(
        "--no-standardize",
        action="store_true",
        help="Do not z-score s/x before atan2.",
    )
    p.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip per-run diagnostic PNGs.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phase_progression",
    )
    p.add_argument(
        "--show-plots",
        action="store_true",
        help="Call plt.show() (default: save only).",
    )
    return p.parse_args()


def resolve_run_list(args: argparse.Namespace, analysis_root: Path) -> List[str]:
    if args.runs:
        return resolve_explicit_runs(
            args.runs,
            analysis_root=str(analysis_root),
            runs_dir=args.runs_dir,
        )

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = analysis_root / runs_dir
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    all_runs = list_run_dirs(
        str(runs_dir),
        exclude_substrings=args.exclude,
        skip_copy=True,
    )
    if not all_runs:
        raise RuntimeError("No runs matched the selection criteria.")
    if args.run_indices is None:
        return all_runs
    selected = []
    for idx in args.run_indices:
        if idx < 0 or idx >= len(all_runs):
            raise IndexError(
                f"run index {idx} out of range for {len(all_runs)} available runs"
            )
        selected.append(all_runs[idx])
    return selected


def infer_study_group(run_path: str, tag: Optional[str] = None) -> str:
    """Map a run path/tag to Study1_visible_s or Study2_hidden_s."""
    hay = (run_path + " " + (tag or "")).lower().replace("\\", "/")
    # Explicit group names already present
    if STUDY1.lower() in hay or "study1_visible" in hay:
        return STUDY1
    if STUDY2.lower() in hay or "study2_hidden" in hay:
        return STUDY2
    # Path / tag heuristics from the plan
    if "/study1" in hay or "study1" in hay or "pknown" in hay:
        return STUDY1
    if "/study2" in hay or "study2" in hay or "punkn" in hay:
        return STUDY2
    raise ValueError(
        f"Could not infer study group for run '{run_path}' "
        f"(tag={tag!r}). Pass --study-groups explicitly."
    )


def normalize_group_label(label: str) -> str:
    low = label.strip().lower()
    if low in {STUDY1.lower(), "study1", "visible", "visible_s", "pknown"}:
        return STUDY1
    if low in {STUDY2.lower(), "study2", "hidden", "hidden_s", "punkn"}:
        return STUDY2
    if "study1" in low or "visible" in low or "pknown" in low:
        return STUDY1
    if "study2" in low or "hidden" in low or "punkn" in low:
        return STUDY2
    return label


def _strip_by_study(
    ax,
    table: pd.DataFrame,
    metric: str,
    title: str,
    colors: dict,
) -> None:
    groups = [STUDY1, STUDY2]
    present = [g for g in groups if (table["group"] == g).any()]
    if not present:
        present = list(table["group"].astype(str).unique())
    rng = np.random.default_rng(0)
    for i, g in enumerate(present):
        y = table.loc[table["group"] == g, metric].to_numpy(dtype=float)
        x = rng.normal(i, 0.05, size=len(y))
        ax.scatter(
            x,
            y,
            s=50,
            alpha=0.85,
            color=colors.get(g, f"C{i}"),
            zorder=3,
            label=g,
        )
        if len(y):
            ax.hlines(
                np.nanmean(y),
                i - 0.25,
                i + 0.25,
                colors=colors.get(g, f"C{i}"),
                lw=1.5,
                zorder=2,
            )
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(
        [
            (
                "Study1\nvisible s"
                if g == STUDY1
                else "Study2\nhidden s"
                if g == STUDY2
                else g
            )
            for g in present
        ],
        fontsize=9,
    )
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xlim(-0.6, len(present) - 0.4)


def plot_study_contrast(
    metrics_by_run: Sequence[Dict],
    table: pd.DataFrame,
    *,
    out_path: Path,
    show: bool,
    x_note: str = "x = A / total resource encounters",
) -> None:
    """Four-panel Study1 vs Study2 contrast: unwrapped θ, D, D_smooth, N_rot_smooth."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    colors = {STUDY1: "tab:blue", STUDY2: "tab:orange"}
    labels_drawn = set()

    # Panel 1: unwrapped θ/2π vs step (smoothed trajectory when available)
    ax = axes[0]
    for m in metrics_by_run:
        group = m["group"]
        th = m.get("theta_unwrapped_smooth")
        if th is None or len(th) == 0:
            th = m.get("theta_unwrapped", np.array([]))
        tt = m["t"]
        if len(th) == 0 or len(tt) == 0:
            continue
        n = min(len(th), len(tt))
        label = group if group not in labels_drawn else None
        ax.plot(
            tt[:n],
            th[:n] / (2 * np.pi),
            color=colors.get(group, "0.4"),
            lw=0.9,
            alpha=0.7,
            label=label,
        )
        if label:
            labels_drawn.add(group)
    ax.set_xlabel("step")
    ax.set_ylabel("θ / 2π (turns)")
    ax.set_title("Unwrapped phase")
    if labels_drawn:
        ax.legend(fontsize=8, loc="best")

    # Panels 2–4: raw D (sensitivity) + smoothed D / N_rot (primary)
    _strip_by_study(axes[1], table, "D", r"$D$ (raw) by study", colors)
    _strip_by_study(axes[2], table, "D_smooth", r"$D_{\mathrm{smooth}}$ by study", colors)
    _strip_by_study(
        axes[3], table, "N_rot_smooth", r"$N_{\mathrm{rot,smooth}}$ by study", colors
    )

    fig.suptitle(
        f"Study1 (visible s) vs Study2 (hidden s) phase progression\n({x_note})",
        y=1.05,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    args = parse_args()
    analysis_root = Path(args.analysis_root).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = analysis_root / out_dir
    out_dir = out_dir.resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    runs = resolve_run_list(args, analysis_root)
    if not runs:
        raise RuntimeError("No runs selected.")

    tags = list(args.condition_tags) if args.condition_tags else [None] * len(runs)
    if len(tags) != len(runs):
        raise ValueError(
            f"--condition-tags count ({len(tags)}) must match runs ({len(runs)})"
        )

    if args.study_groups is not None:
        if len(args.study_groups) != len(runs):
            raise ValueError(
                f"--study-groups count ({len(args.study_groups)}) must match runs ({len(runs)})"
            )
        groups = [normalize_group_label(g) for g in args.study_groups]
    else:
        groups = [infer_study_group(r, tag=t) for r, t in zip(runs, tags)]

    last_n = None if args.last_n is None or args.last_n <= 0 else int(args.last_n)
    start_after = (
        None
        if args.start_after is None or args.start_after <= 0
        else int(args.start_after)
    )
    standardize = not args.no_standardize
    x_mode = args.x_mode
    detrend = args.detrend
    detrend_rolling_w = args.detrend_rolling_w
    x_label = None

    print(
        "NOTE: phase x-axis default is A/total resource encounters "
        f"(--x-mode {x_mode}); detrend={detrend}; "
        f"start_after={start_after}; last_n={last_n}; "
        f"smooth_w={args.smooth_w}."
    )

    rows = []
    metrics_by_run: List[Dict] = []

    for run_dir, tag, group in zip(runs, tags, groups):
        run_name = Path(run_dir).name
        print(f"Processing {run_name}  [{group}]")
        t, s, x, x_label = load_s_x_series(run_dir, x_mode=x_mode)
        m = phase_progression_metrics(
            s,
            x,
            t=t,
            last_n=last_n,
            start_after=start_after,
            smooth_w=args.smooth_w,
            standardize=standardize,
            detrend=detrend,
            detrend_rolling_w=detrend_rolling_w,
            n_surr=args.n_surr,
        )
        m["group"] = group
        m["run"] = run_name
        m["run_dir"] = run_dir
        metrics_by_run.append(m)

        row = {
            "run": run_name,
            "run_dir": run_dir,
            "group": group,
            "condition_tag": tag,
            "x_mode": x_mode,
            "x_label": x_label,
            "detrend": detrend,
            "detrend_rolling_w": detrend_rolling_w,
            "n": m["n"],
            "D": m["D"],
            "N_rot": m["N_rot"],
            "D_smooth": m["D_smooth"],
            "N_rot_smooth": m["N_rot_smooth"],
            "smooth_w": m["smooth_w"],
            "last_n": m["last_n"],
            "start_after": m.get("start_after"),
        }
        if args.n_surr and args.n_surr > 0:
            row["D_surr_mean"] = m.get("D_surr_mean")
            row["D_surr_p95"] = m.get("D_surr_p95")
            row["D_surr_p"] = m.get("D_surr_p")
        rows.append(row)

        if not args.no_diagnostics:
            diag_path = plots_dir / f"{run_name}_phase_diag.png"
            plot_phase_diagnostics(
                t,
                s,
                x,
                last_n=last_n,
                start_after=start_after,
                smooth_w=args.smooth_w,
                standardize=standardize,
                detrend=detrend,
                detrend_rolling_w=detrend_rolling_w,
                title=f"{run_name} ({group})\nx = {x_label}; detrend={detrend}",
                out_path=diag_path,
                show=False,
            )
            print(f"  wrote {diag_path}")

    table = pd.DataFrame(rows)
    metrics_csv = out_dir / "phase_metrics.csv"
    table.to_csv(metrics_csv, index=False)
    print(f"Wrote {metrics_csv}")

    contrast_path = plots_dir / "study1_vs_study2_phase_contrast.png"
    window_note = (
        f"t>={start_after}" if start_after is not None else "full series"
    )
    if last_n is not None:
        window_note += f", last {last_n}"
    plot_study_contrast(
        metrics_by_run,
        table,
        out_path=contrast_path,
        show=args.show_plots,
        x_note=f"x = {x_label or x_mode}; detrend={detrend}; {window_note}",
    )
    print(f"Wrote {contrast_path}")

    note_path = out_dir / "X_AXIS_NOTE.txt"
    note_path.write_text(
        "Phase-progression x-axis / preprocessing\n"
        "========================================\n"
        f"x_mode: {x_mode}\n"
        f"definition: {x_label}\n"
        "s: Global_average_punishment_level\n"
        f"detrend: {detrend}\n"
        f"detrend_rolling_w: {detrend_rolling_w}\n"
        f"start_after: {start_after}\n"
        f"last_n: {last_n}\n"
        f"smooth_w: {args.smooth_w}\n"
        "\n"
        "Order: start-after (t>=N) → optional last-N → detrend(s,x) → "
        "smooth → z-score → angle.\n"
        "Default start_after=40000 (all steps from 40000 onward).\n"
        "Default smooth_w=10; contrast panels include raw D and D_smooth.\n"
        "Default detrend=linear so D/N_rot emphasize cycling rather than\n"
        "secular walks. Use --detrend none to keep drift.\n"
        "\n"
        "Default x is A/total (resource A share), not raw A.\n"
        "Use --x-mode raw_a for absolute Mean_mean_a_encounters.\n",
        encoding="utf-8",
    )
    print(f"Wrote {note_path}")

    print("\nSummary:")
    print(table[["run", "group", "D_smooth", "N_rot_smooth", "n"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
