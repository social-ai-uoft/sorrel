#!/usr/bin/env python3
"""Study1 vs Study2 mean-flow + per-cell visit time-series plots.

NOTE — x-axis definition (default):
  s = Global_average_punishment_level
  x = Mean_mean_a_encounters / sum_r Mean_mean_r_encounters
      (resource A share of total resources extracted; NOT raw A)
  Override with --x-mode raw_a if absolute A intensity is needed.

Loads (s, x) from resolved run folders, groups into Study1_visible_s /
Study2_hidden_s panels, and writes the contrast PNG via plot_mean_flow_visit_ts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from plot_mean_flow_visit_ts_standalone import plot_mean_flow_visit_ts
from system_analysis_utils import list_run_dirs, load_s_x_series, resolve_explicit_runs

STUDY1 = "Study1_visible_s"
STUDY2 = "Study2_hidden_s"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mean-flow + visit-ts mosaic for Study1 vs Study2 runs."
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
        help="Optional per-run labels / study-group hints.",
    )
    p.add_argument(
        "--study-groups",
        nargs="*",
        default=None,
        help=f"Explicit group labels per run. Preferred: {STUDY1} / {STUDY2}.",
    )
    p.add_argument(
        "--last-n",
        type=int,
        default=20_000,
        help="Keep only the last N steps (default 20000). Set 0 for full series.",
    )
    p.add_argument(
        "--x-mode",
        type=str,
        default="a_share",
        choices=["a_share", "raw_a"],
        help=(
            "Mean-flow x observable. Default a_share = A/total resource "
            "encounters (preferred). raw_a = absolute Mean_mean_a_encounters."
        ),
    )
    p.add_argument("--smooth", type=int, default=40, dest="smooth_window")
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--n-time-bins", type=int, default=5000)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument(
        "--auto-lim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Zoom x/s axes to occupied data mass and re-bin there (default). "
            "Use --no-auto-lim for the fixed (0,1)×(-0.5,9.5) box."
        ),
    )
    p.add_argument("--x-min", type=float, default=None)
    p.add_argument("--x-max", type=float, default=None)
    p.add_argument("--s-min", type=float, default=None)
    p.add_argument("--s-max", type=float, default=None)
    p.add_argument(
        "--lim-expand",
        type=float,
        default=2.25,
        help="With --auto-lim, scale axis window about its center (2.25 = two 50% zoom-outs).",
    )
    p.add_argument("--x-scale", type=float, default=10.0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--x-label",
        type=str,
        default=None,
        help="Override x-axis label (default depends on --x-mode).",
    )
    p.add_argument(
        "--s-label",
        type=str,
        default=r"$s$ (punishment)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mean_flow_visit_ts",
    )
    p.add_argument(
        "--show-plots",
        action="store_true",
        help="Accepted for launcher compatibility (figure is always saved).",
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
    hay = (run_path + " " + (tag or "")).lower().replace("\\", "/")
    if STUDY1.lower() in hay or "study1_visible" in hay:
        return STUDY1
    if STUDY2.lower() in hay or "study2_hidden" in hay:
        return STUDY2
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


def panel_display_name(group: str) -> str:
    if group == STUDY1:
        return "Study1 (visible s)"
    if group == STUDY2:
        return "Study2 (hidden s)"
    return group


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
    x_mode = args.x_mode
    x_label_resolved = None

    print(
        "NOTE: mean-flow x-axis default is A/total resource encounters "
        f"(--x-mode {x_mode})."
    )

    panels: Dict[str, list] = {}
    meta_rows = []
    # Preserve Study1 then Study2 panel order when both present.
    panel_order = []
    for run_dir, tag, group in zip(runs, tags, groups):
        run_name = Path(run_dir).name
        print(f"Loading {run_name}  [{group}]")
        t, s, x, x_label_resolved = load_s_x_series(run_dir, x_mode=x_mode)
        display = panel_display_name(group)
        if display not in panels:
            panels[display] = []
            panel_order.append(display)
        panels[display].append({"t": t, "x": x, "s": s})
        meta_rows.append(
            {
                "run": run_name,
                "run_dir": run_dir,
                "group": group,
                "panel": display,
                "condition_tag": tag,
                "x_mode": x_mode,
                "x_label": x_label_resolved,
                "n_steps": int(len(t)),
                "last_n": last_n,
            }
        )

    if args.x_label is not None:
        plot_x_label = args.x_label
    elif x_mode == "raw_a":
        plot_x_label = r"$x$ (raw resource A)"
    else:
        plot_x_label = r"$x$ (A / total resources)"

    ordered_panels = {name: panels[name] for name in panel_order}
    out_png = plots_dir / "study1_vs_study2_mean_flow_visit_ts.png"
    print(f"Plotting {len(ordered_panels)} panel(s) -> {out_png}")

    auto_lim = bool(args.auto_lim)
    lims_set = any(
        v is not None for v in (args.x_min, args.x_max, args.s_min, args.s_max)
    )
    if lims_set:
        auto_lim = False
    x_lim = (
        (
            float(args.x_min if args.x_min is not None else 0.0),
            float(args.x_max if args.x_max is not None else 1.0),
        )
        if not auto_lim
        else None
    )
    s_lim = (
        (
            float(args.s_min if args.s_min is not None else -0.5),
            float(args.s_max if args.s_max is not None else 9.5),
        )
        if not auto_lim
        else None
    )
    print(
        "Axis mode: "
        + (
            "auto-zoom to occupied (x,s) mass"
            if auto_lim
            else f"fixed x_lim={x_lim} s_lim={s_lim}"
        )
    )

    plot_mean_flow_visit_ts(
        ordered_panels,
        out_png,
        n_bins=args.n_bins,
        smooth_window=args.smooth_window,
        last_n_steps=last_n,
        n_time_bins=args.n_time_bins,
        min_count=args.min_count,
        x_lim=x_lim,
        s_lim=s_lim,
        auto_lim=auto_lim,
        lim_expand=float(args.lim_expand),
        x_scale=args.x_scale,
        x_label=plot_x_label,
        s_label=args.s_label,
        dpi=args.dpi,
        title=(
            "Study1 (visible s) vs Study2 (hidden s)\n"
            f"Mean flow + per-cell visit time series\n"
            f"(x = {x_label_resolved or x_mode})"
        ),
    )
    print(f"Wrote {out_png}")
    meta_csv = out_dir / "mean_flow_run_manifest.csv"
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)
    print(f"Wrote {meta_csv}")

    note_path = out_dir / "X_AXIS_NOTE.txt"
    note_path.write_text(
        "Mean-flow / visit-ts x-axis\n"
        f"Axis zoom: {'auto (occupied mass)' if auto_lim else 'fixed'}\n"
        f"x_lim={x_lim} s_lim={s_lim} (None => chosen inside plotter)\n"
        "===========================\n"
        f"x_mode: {x_mode}\n"
        f"definition: {x_label_resolved}\n"
        "s: Global_average_punishment_level\n"
        "\n"
        "Default is A/total (resource A share of total extraction), not raw A.\n"
        "Use --x-mode raw_a for absolute Mean_mean_a_encounters.\n",
        encoding="utf-8",
    )
    print(f"Wrote {note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
