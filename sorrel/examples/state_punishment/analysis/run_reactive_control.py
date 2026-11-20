#!/usr/bin/env python3
"""Run reactive-control system analysis over state-punishment runs.

Extracted from analysis_v2.ipynb. Depends on system_analysis_utils.py.

Example (from this directory):

    python run_reactive_control.py

    python run_reactive_control.py \\
        --runs-dir res/runs_bandit_study2_v3_twoArms \\
        --exclude k5 seed3 seed4 \\
        --output-dir outputs/reactive_control

    # Optional truncate (default is full series):
    #   --xlim 60000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

import pandas as pd

from system_analysis_utils import (
    annotate_run_metadata,
    list_run_dirs,
    resolve_explicit_runs,
    system_analysis_reactive_control,
)

# Preserved from the notebook for optional reuse / comparison.
FS_STUDY1 = [
    "res/runs_debug7/00_cpc_iqn_lstm_cpc_slow_v2_prob_ext_charm_immed_silly_pnoobs_pothnoobs_noapp_reprten_n1_min10000_start10000_ten10000_none_iqn_preset_sp_r0.010_v4_m10_cvTrue_meTrue_10a_pknown_sknown_20260217-121922",
    "res/runs_debug7/01_cpc_iqn_lstm_cpc_slow_v2_prob_ext_charm_immed_silly_pnoobs_pothnoobs_noapp_reprten_n1_min10000_start10000_ten10000_none_iqn_preset_sp_r0.010_v4_m10_cvTrue_meTrue_10a_pknown_sknown_20260217-121824",
    "res/runs_debug7/00_cpc_ppo_lstm_cpc_slow_v2_prob_ext_charm_immed_silly_pnoobs_pothnoobs_noapp_reprten_n1_min10000_start10000_ten10000_none_ppo_lstm_cpc_preset_sp_r0.010_v4_m10_cvTrue_meTrue_10a_pknown_sknown_20260217-122249",
    "res/runs_debug7/01_cpc_ppo_lstm_cpc_slow_v2_prob_ext_charm_immed_silly_pnoobs_pothnoobs_noapp_reprten_n1_min10000_start10000_ten10000_none_ppo_lstm_cpc_preset_sp_r0.010_v4_m10_cvTrue_meTrue_10a_pknown_sknown_20260217-122210",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reactive-control analysis over state-punishment runs."
    )
    parser.add_argument(
        "--analysis-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory that contains the res/ folder (default: this script's dir).",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help=(
            "Explicit run directories or basenames. When set, overrides "
            "--runs-dir discovery / --use-study1."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="res/runs_bandit_study2_v3_twoArms",
        help="Relative (to analysis-root) or absolute path to run folders.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["k5", "seed3", "seed4"],
        help="Exclude run names containing any of these substrings.",
    )
    parser.add_argument(
        "--use-study1",
        action="store_true",
        help="Use the hardcoded FS_STUDY1 run list instead of --runs-dir.",
    )
    parser.add_argument(
        "--xlim",
        type=int,
        default=None,
        help="Truncate each series to the first N steps (default: use full series).",
    )
    parser.add_argument("--smooth-social-harm-window", type=int, default=5)
    parser.add_argument("--smooth-punishment-window", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--start-timepoint", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--no-controls",
        action="store_true",
        help="Disable include_controls (default: controls on).",
    )
    parser.add_argument(
        "--no-rate",
        action="store_true",
        help="Disable use_rate (default: rate on).",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable standardization in regressions.",
    )
    parser.add_argument(
        "--no-recent-avg-harm",
        action="store_true",
        help="Use current harm for panel 2 instead of recent average harm.",
    )
    parser.add_argument("--panel2-recent-avg-window", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reactive_control",
        help="Where to write concatenated results CSV and per-run plots.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Call plt.show() for each run (default: save only, non-interactive).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation entirely.",
    )
    return parser.parse_args()


def resolve_run_list(args: argparse.Namespace, analysis_root: Path) -> list[str]:
    if args.runs:
        return resolve_explicit_runs(
            args.runs,
            analysis_root=str(analysis_root),
            runs_dir=args.runs_dir,
        )

    if args.use_study1:
        return [str((analysis_root / p).resolve()) for p in FS_STUDY1]

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = analysis_root / runs_dir
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    return list_run_dirs(
        str(runs_dir),
        exclude_substrings=args.exclude,
        skip_copy=True,
    )


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root).resolve()
    os.chdir(analysis_root)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = analysis_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not args.show_plots:
        matplotlib.use("Agg")

    fs = resolve_run_list(args, analysis_root)
    if not fs:
        raise RuntimeError("No runs matched the selection criteria.")

    print(f"Analysis root: {analysis_root}")
    print(f"Selected {len(fs)} runs:")
    for path in fs:
        print(f"  - {path}")

    lims = [args.xlim for _ in range(len(fs))]
    all_results = []

    for idx, f in enumerate(fs):
        run_name = Path(f).name
        print(f"\n[{idx + 1}/{len(fs)}] {run_name}")

        save_plot_path = None
        if not args.no_plots:
            save_plot_path = str(plots_dir / f"{run_name}_reactive_control.png")

        results_df, _data_df = system_analysis_reactive_control(
            f=f,
            include_controls=not args.no_controls,
            use_rate=not args.no_rate,
            smooth_social_harm_window=args.smooth_social_harm_window,
            smooth_punishment_window=args.smooth_punishment_window,
            window_size=args.window_size,
            step_size=args.step_size,
            lag=args.lag,
            start_timepoint=args.start_timepoint,
            xlim=lims[idx],
            alpha=args.alpha,
            standardize=not args.no_standardize,
            panel2_use_recent_avg_harm=not args.no_recent_avg_harm,
            panel2_recent_avg_window=args.panel2_recent_avg_window,
            make_plot=not args.no_plots,
            save_plot_path=save_plot_path,
            show_plot=args.show_plots,
        )

        results_df = annotate_run_metadata(results_df, f)
        all_results.append(results_df)

    all_results_df = pd.concat(all_results, ignore_index=True)
    out_csv = output_dir / "all_results_df.csv"
    all_results_df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(all_results_df)} rows -> {out_csv}")
    print(all_results_df.groupby(["study_number", "model_type", "use_cpc", "test"]).size())


if __name__ == "__main__":
    main()
