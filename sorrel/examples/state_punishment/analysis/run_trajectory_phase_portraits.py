#!/usr/bin/env python3
"""Plot punishment / harm / vote / resource trajectories and phase portraits.

Covers the analysis_v2.ipynb section
"plot state punishment, social harm received, and actions"
up to (but not including) spectral band analysis, and adds 2D + 3D
phase portraits (time/episode as the third axis).

Default phase-portrait y-axis is A/total resource share (not raw A encounters).

Depends on system_analysis_utils.py.

Example (from this directory):

    python run_trajectory_phase_portraits.py

    python run_trajectory_phase_portraits.py \\
        --runs-dir res/runs_bandit_study2_v3_twoArms \\
        --run-indices 0 1 \\
        --condition-tags "Punishment level Visible" "Punishment level Hidden"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend
from statsmodels.graphics.tsaplots import plot_acf

from system_analysis_utils import (
    _prepare_xy,
    list_run_dirs,
    load_runs_trajectories,
    phase_portrait_xy,
    phase_portrait_xyt,
    resolve_explicit_runs,
)

FS_STUDY1 = [
    "res/runs_debug7/00_cpc_iqn_lstm_cpc_slow_v2_prob_ext_charm_immed_silly_pnoobs_pothnoobs_noapp_reprten_n1_min10000_start10000_ten10000_none_iqn_preset_sp_r0.010_v4_m10_cvTrue_meTrue_10a_pknown_sknown_20260217-121922",
    "res/runs_debug7/01_cpc_iqn_lstm_cpc_slow_v2_prob_ext_charm_immed_silly_pnoobs_pothnoobs_noapp_reprten_n1_min10000_start10000_ten10000_none_iqn_preset_sp_r0.010_v4_m10_cvTrue_meTrue_10a_pknown_sknown_20260217-121824",
    "res/runs_debug7/00_cpc_ppo_lstm_cpc_slow_v2_prob_ext_charm_immed_silly_pnoobs_pothnoobs_noapp_reprten_n1_min10000_start10000_ten10000_none_ppo_lstm_cpc_preset_sp_r0.010_v4_m10_cvTrue_meTrue_10a_pknown_sknown_20260217-122249",
    "res/runs_debug7/01_cpc_ppo_lstm_cpc_slow_v2_prob_ext_charm_immed_silly_pnoobs_pothnoobs_noapp_reprten_n1_min10000_start10000_ten10000_none_ppo_lstm_cpc_preset_sp_r0.010_v4_m10_cvTrue_meTrue_10a_pknown_sknown_20260217-122210",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trajectory plots + 2D/3D phase portraits for state punishment runs."
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
        help=(
            "Explicit run directories or basenames. When set, overrides "
            "--runs-dir discovery / --use-study1 / --run-indices."
        ),
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
        default=["k5", "seed3", "seed4"],
        help="Exclude run names containing any of these substrings.",
    )
    p.add_argument(
        "--use-study1",
        action="store_true",
        help="Use the hardcoded FS_STUDY1 run list instead of --runs-dir.",
    )
    p.add_argument(
        "--run-indices",
        nargs="*",
        type=int,
        default=[0, 1],
        help="Which discovered runs to plot (ignored when --runs is set).",
    )
    p.add_argument(
        "--condition-tags",
        nargs="*",
        default=["Punishment level Visible", "Punishment level Hidden"],
        help="Legend labels for each selected run (must match selected run count).",
    )
    p.add_argument("--end", type=int, default=None, help="Optional end step for plots.")
    p.add_argument("--resource-wsize", type=int, default=2)
    p.add_argument("--metric-wsize", type=int, default=10)
    p.add_argument(
        "--phase-last-n",
        type=int,
        default=40_000,
        help="Use only the last N steps for phase 2D/3D portraits (default 40000). 0 = full series.",
    )
    p.add_argument(
        "--phase-windows",
        nargs="*",
        type=int,
        default=[2, 10],
        help="Rolling windows used for phase portraits (default: 2, 10).",
    )
    p.add_argument(
        "--phase-y",
        type=str,
        default="resource_A_share",
        choices=[
            "resource_A_share",
            "resource_A",
            "resource_B",
            "resource_C",
            "resource_D",
            "resource_E",
            "social_harm_received",
            "resource_total",
        ],
        help=(
            "Y-axis series for phase portraits (X is always punishment level). "
            "Default resource_A_share = A / total resource encounters."
        ),
    )
    p.add_argument(
        "--phase-2d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write 2D phase portraits (default on). Disable with --no-phase-2d."
        ),
    )
    p.add_argument(
        "--phase-run-index",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Indices into the selected run list for phase portraits. "
            "Default: all selected runs."
        ),
    )
    p.add_argument("--phase-step", type=int, default=1, help="Subsample step for portraits.")
    p.add_argument("--phase-arrow-every", type=int, default=50)
    p.add_argument(
        "--acf-lags",
        type=int,
        default=40000,
        help="ACF lags for detrended resource A (set 0 to skip).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/trajectory_phase_portraits",
    )
    p.add_argument(
        "--show-plots",
        action="store_true",
        help="Call plt.show() (default: save only, non-interactive).",
    )
    p.add_argument(
        "--no-resource-plot",
        action="store_true",
        help="Skip per-resource trajectory figure.",
    )
    p.add_argument(
        "--no-metric-plot",
        action="store_true",
        help="Skip punishment/vote/reward trajectory figure.",
    )
    p.add_argument(
        "--no-phase",
        action="store_true",
        help="Skip phase portrait figures.",
    )
    return p.parse_args()


def resolve_run_list(
    args: argparse.Namespace, analysis_root: Path
) -> Tuple[List[str], List[str]]:
    if args.runs:
        selected = resolve_explicit_runs(
            args.runs,
            analysis_root=str(analysis_root),
            runs_dir=args.runs_dir,
        )
        return selected, selected

    if args.use_study1:
        all_runs = [str((analysis_root / p).resolve()) for p in FS_STUDY1]
    else:
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

    selected = []
    for idx in args.run_indices:
        if idx < 0 or idx >= len(all_runs):
            raise IndexError(
                f"run index {idx} out of range for {len(all_runs)} available runs"
            )
        selected.append(all_runs[idx])
    return selected, all_runs


def save_or_show(fig, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_resource_trajectories(
    traj,
    condition_tags: Sequence[str],
    wsize: int,
    end: Optional[int],
    output_path: Path,
    show: bool,
) -> None:
    resource_labels = [
        "A / total",
        "B / total",
        "C / total",
        "D / total",
        "E / total",
    ]
    resource_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    resource_keys = [
        "resource_A",
        "resource_B",
        "resource_C",
        "resource_D",
        "resource_E",
    ]
    n_runs = len(condition_tags)

    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    for i, (label, color, key) in enumerate(
        zip(resource_labels, resource_colors, resource_keys)
    ):
        for count in range(n_runs):
            total = traj["resource_total"][count].replace(0, np.nan)
            series = traj[key][count] / total
            axs[i].plot(
                series[:end].rolling(window=wsize).mean(),
                label=f"{label}, {condition_tags[count]}",
                color=color if n_runs == 1 else None,
                linestyle="-" if count == 0 else "--",
            )
            print(f"{label} / {condition_tags[count]} mean[-2000:]:", series[-2000:end].mean())
        axs[i].set_ylabel(label)
        axs[i].set_ylim(0, 1)
        axs[i].legend()

    axs[-1].set_xlabel("Step")
    fig.tight_layout()
    save_or_show(fig, output_path, show)


def plot_metric_trajectories(
    traj,
    condition_tags: Sequence[str],
    wsize: int,
    end: Optional[int],
    output_path: Path,
    show: bool,
) -> None:
    metric_labels = [
        ("Punishment level", "tab:blue", "state_punishment_level"),
        ("Vote increase", "tab:green", "vote_increase"),
        ("Vote decrease", "tab:red", "vote_decrease"),
        ("Reward", "tab:purple", "reward_series"),
    ]
    n_runs = len(condition_tags)
    fig, axs = plt.subplots(len(metric_labels), 1, figsize=(14, 24), sharex=True)

    for i, (label, color, key) in enumerate(metric_labels):
        for count in range(n_runs):
            series = traj[key][count]
            axs[i].plot(
                series[:end].rolling(window=wsize).mean(),
                label=f"{label}, {condition_tags[count]}",
                color=color if n_runs == 1 else None,
                linestyle="-" if count == 0 else "--",
                alpha=0.5,
            )
            if label in ["Vote increase", "Vote decrease"]:
                print(label, condition_tags[count], series[-10000:].mean())
            if label == "Reward":
                print("Reward", condition_tags[count], series[-10000:].mean())

        axs[i].set_ylabel(label, fontsize=20)
        if label == "Punishment level":
            axs[i].set_ylim(0, 1)
            axs[i].axhline(0.1, color="gray", linestyle="--", linewidth=1)
            axs[i].axhline(0.2, color="gray", linestyle="--", linewidth=1)
        if label in ["Vote increase", "Vote decrease"]:
            axs[i].axhline(2, color="gray", linestyle="--", linewidth=1)
        axs[i].legend()

    axs[-1].set_xlabel("Step")
    fig.tight_layout()
    save_or_show(fig, output_path, show)


def plot_acf_resource_a(
    series: pd.Series,
    lags: int,
    output_path: Path,
    show: bool,
) -> None:
    signal_d = detrend(np.asarray(series, dtype=float))
    # statsmodels requires lags < nobs; clamp for short runs.
    max_lags = max(1, len(signal_d) - 1)
    lags_use = int(min(max(1, lags), max_lags))
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    plot_acf(signal_d, lags=lags_use, ax=ax)
    ax.set_title(f"ACF of detrended A / total share (lags={lags_use})")
    fig.tight_layout()
    save_or_show(fig, output_path, show)


def _phase_slice_bounds(
    s: pd.Series,
    y: pd.Series,
    t: pd.Series,
    *,
    window: int,
    start: Optional[int],
    end: Optional[int],
    step: int,
) -> Tuple[float, float, float, float, float, float]:
    s_s = s.rolling(window=window).mean()
    y_s = y.rolling(window=window).mean()
    xs, ys, ts = _prepare_xy(s_s, y_s, start=start, end=end, step=step, time=t)
    return (
        float(np.nanmin(xs)),
        float(np.nanmax(xs)),
        float(np.nanmin(ys)),
        float(np.nanmax(ys)),
        float(np.nanmin(ts)),
        float(np.nanmax(ts)),
    )


def _pad_lim(lo: float, hi: float, pad_frac: float = 0.05) -> Tuple[float, float]:
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        pad = 0.05 if abs(hi) < 1 else 0.05 * abs(hi)
        return lo - pad, hi + pad
    pad = pad_frac * span
    return lo - pad, hi + pad


def compute_shared_phase_limits(
    traj,
    run_indices: Sequence[int],
    y_key: str,
    windows: Sequence[int],
    end: Optional[int],
    step: int,
    last_n: Optional[int],
) -> dict:
    """Union axis ranges across runs (per smooth window) for fair Study1/2 comparison."""
    shared: dict = {}
    for w in windows:
        xs0, xs1, ys0, ys1, ts0, ts1 = [], [], [], [], [], []
        for idx in run_indices:
            s = traj["state_punishment_level"][idx]
            y = traj[y_key][idx]
            t = traj["steps"][idx]
            n = int(len(s))
            start = None
            end_use = end
            if last_n is not None and int(last_n) > 0 and n > int(last_n):
                start = n - int(last_n)
                end_use = None
            a0, a1, b0, b1, c0, c1 = _phase_slice_bounds(
                s, y, t, window=int(w), start=start, end=end_use, step=step
            )
            xs0.append(a0); xs1.append(a1)
            ys0.append(b0); ys1.append(b1)
            ts0.append(c0); ts1.append(c1)
        shared[int(w)] = {
            "xlim": _pad_lim(min(xs0), max(xs1)),
            "ylim": _pad_lim(min(ys0), max(ys1)),
            "zlim": _pad_lim(min(ts0), max(ts1)),
        }
    return shared


def plot_phase_portraits(
    traj,
    run_idx: int,
    condition_tag: str,
    y_key: str,
    windows: Sequence[int],
    end: Optional[int],
    step: int,
    arrow_every: int,
    output_dir: Path,
    show: bool,
    make_2d: bool = False,
    last_n: Optional[int] = 40_000,
    shared_limits: Optional[dict] = None,
) -> None:
    s = traj["state_punishment_level"][run_idx]
    y = traj[y_key][run_idx]
    t = traj["steps"][run_idx]

    y_name_map = {
        "resource_A_share": "x (A / total resources)",
        "resource_A": "resource A encounters",
        "resource_B": "resource B encounters",
        "resource_C": "resource C encounters",
        "resource_D": "resource D encounters",
        "resource_E": "resource E encounters",
        "social_harm_received": "social harm received",
        "resource_total": "total resource encounters",
    }
    y_name = y_name_map.get(y_key, y_key)

    # Prefer last-N window for portraits; fall back to legacy [:end] if last_n off.
    n = int(len(s))
    start = None
    end_use = end
    last_note = "all steps"
    if last_n is not None and int(last_n) > 0:
        ln = int(last_n)
        if n > ln:
            start = n - ln
        end_use = None
        last_note = f"last {min(n, ln):,} steps"

    for w in windows:
        s_smooth = s.rolling(window=w).mean()
        y_smooth = y.rolling(window=w).mean()
        tag = f"{condition_tag.replace(' ', '_')}_w{w}"
        lims = (shared_limits or {}).get(int(w), {})
        xlim = lims.get("xlim")
        ylim = lims.get("ylim")
        zlim = lims.get("zlim")

        if make_2d:
            phase_portrait_xy(
                s_smooth,
                y_smooth,
                x_name="punishment strictness s_t",
                y_name=y_name,
                title=f"2D phase portrait ({condition_tag}, roll={w}; {last_note})",
                start=start,
                end=end_use,
                step=step,
                arrows=False,
                arrow_every=arrow_every,
                color_by_time=True,
                connect=True,
                show_points=False,
                line_width=1.4,
                line_alpha=0.9,
                xlim=xlim,
                ylim=ylim,
                save_path=str(output_dir / f"phase2d_{tag}.png"),
                show=show,
            )

        phase_portrait_xyt(
            s_smooth,
            y_smooth,
            time=t,
            x_name="punishment strictness s_t",
            y_name=y_name,
            t_name="t (step / episode)",
            title=f"3D phase portrait ({condition_tag}, roll={w}; {last_note})",
            start=start,
            end=end_use,
            step=step,
            arrows=False,
            arrow_every=arrow_every,
            connect=True,
            show_points=False,
            line_width=1.4,
            line_alpha=0.9,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            save_path=str(output_dir / f"phase3d_{tag}.png"),
            show=show,
        )


_PHASE_Y_NAMES = {
    "resource_A_share": "x (A / total resources)",
    "resource_A": "resource A encounters",
    "resource_B": "resource B encounters",
    "resource_C": "resource C encounters",
    "resource_D": "resource D encounters",
    "resource_E": "resource E encounters",
    "social_harm_received": "social harm received",
    "resource_total": "total resource encounters",
}

# Discrete level where resource A net value first goes negative
# (state_system predefined schedule: s=4 "A smaller than 0";
#  a_value=20, punishment_magnitude=25 → NV_A = 20 - 0.85*25 = -1.25).
# Logged s_t = prob ∈ [0, 1] with level = round(prob * (num_steps - 1)).
_A_NET_NEGATIVE_LEVEL = 4
_NUM_PUNISHMENT_STEPS = 10
_A_NET_NEGATIVE_S = _A_NET_NEGATIVE_LEVEL / (_NUM_PUNISHMENT_STEPS - 1)  # 4/9 ≈ 0.444


def _phase_slice_start_end(
    n: int, end: Optional[int], last_n: Optional[int]
) -> Tuple[Optional[int], Optional[int], str]:
    start = None
    end_use = end
    last_note = "all steps"
    if last_n is not None and int(last_n) > 0:
        ln = int(last_n)
        if n > ln:
            start = n - ln
        end_use = None
        last_note = f"last {min(n, ln):,} steps"
    return start, end_use, last_note


def plot_phase_comparison(
    traj,
    run_indices: Sequence[int],
    condition_tags: Sequence[str],
    y_key: str,
    windows: Sequence[int],
    end: Optional[int],
    step: int,
    arrow_every: int,
    output_dir: Path,
    show: bool,
    make_2d: bool = True,
    last_n: Optional[int] = 40_000,
    shared_limits: Optional[dict] = None,
) -> None:
    """Side-by-side Study1 | Study2 phase portraits with shared axis limits."""
    if len(run_indices) < 2:
        return
    # Default: first two selected runs (Study1 vs Study2).
    pair = list(run_indices[:2])
    tags = [condition_tags[i] for i in pair]
    y_name = _PHASE_Y_NAMES.get(y_key, y_key)
    x_name = "punishment strictness s_t"

    for w in windows:
        lims = (shared_limits or {}).get(int(w), {})
        xlim = lims.get("xlim")
        ylim = lims.get("ylim")
        zlim = lims.get("zlim")
        panel_data = []
        last_notes = []
        for idx in pair:
            s = traj["state_punishment_level"][idx]
            y = traj[y_key][idx]
            t = traj["steps"][idx]
            start, end_use, last_note = _phase_slice_start_end(int(len(s)), end, last_n)
            last_notes.append(last_note)
            panel_data.append(
                (
                    s.rolling(window=w).mean(),
                    y.rolling(window=w).mean(),
                    t,
                    start,
                    end_use,
                )
            )
        last_note = last_notes[0] if last_notes[0] == last_notes[1] else "last-N window"

        if make_2d:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
            for i, (ax, (s_sm, y_sm, _t, start, end_use), tag) in enumerate(
                zip(axes, panel_data, tags)
            ):
                phase_portrait_xy(
                    s_sm,
                    y_sm,
                    x_name=x_name,
                    y_name=y_name,
                    title=tag.replace("_", " "),
                    start=start,
                    end=end_use,
                    step=step,
                    arrows=False,
                    arrow_every=arrow_every,
                    color_by_time=True,
                    connect=True,
                    show_points=False,
                    line_width=1.4,
                    line_alpha=0.9,
                    xlim=xlim,
                    ylim=ylim,
                    save_path=None,
                    show=False,
                    ax=ax,
                )
                # Red line: first level where private net value of A is < 0.
                ax.axvline(
                    _A_NET_NEGATIVE_S,
                    color="red",
                    linewidth=1.8,
                    linestyle="--",
                    zorder=6,
                    label=(
                        rf"$NV_A<0$ (level {_A_NET_NEGATIVE_LEVEL}, "
                        rf"$s={_A_NET_NEGATIVE_S:.3f}$)"
                        if i == 0
                        else None
                    ),
                )
            axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
            fig.suptitle(
                f"2D phase: Study1 vs Study2 (roll={w}; {last_note})",
                y=1.02,
            )
            fig.tight_layout()
            out = output_dir / f"phase2d_compare_w{w}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)

        fig = plt.figure(figsize=(16, 7))
        for i, ((s_sm, y_sm, t, start, end_use), tag) in enumerate(
            zip(panel_data, tags)
        ):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            phase_portrait_xyt(
                s_sm,
                y_sm,
                time=t,
                x_name=x_name,
                y_name=y_name,
                t_name="t (step / episode)",
                title=tag.replace("_", " "),
                start=start,
                end=end_use,
                step=step,
                arrows=False,
                arrow_every=arrow_every,
                connect=True,
                show_points=False,
                line_width=1.4,
                line_alpha=0.9,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                save_path=None,
                show=False,
                ax=ax,
            )
        fig.suptitle(
            f"3D phase: Study1 vs Study2 (roll={w}; {last_note})",
            y=0.98,
        )
        fig.tight_layout()
        out = output_dir / f"phase3d_compare_w{w}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root).resolve()
    os.chdir(analysis_root)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = analysis_root / output_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not args.show_plots:
        matplotlib.use("Agg")

    selected, all_runs = resolve_run_list(args, analysis_root)
    if len(args.condition_tags) != len(selected):
        raise ValueError(
            f"--condition-tags length ({len(args.condition_tags)}) "
            f"must match selected runs ({len(selected)})"
        )

    print(f"Available runs ({len(all_runs)}):")
    for i, path in enumerate(all_runs):
        mark = "*" if path in selected else " "
        print(f"  {mark} [{i}] {Path(path).name}")
    print(f"\nSelected {len(selected)} runs for plotting.")

    traj = load_runs_trajectories(selected)

    if not args.no_resource_plot:
        plot_resource_trajectories(
            traj,
            condition_tags=args.condition_tags,
            wsize=args.resource_wsize,
            end=args.end,
            output_path=plots_dir / "resource_trajectories.png",
            show=args.show_plots,
        )

    if args.acf_lags and args.acf_lags > 0:
        plot_acf_resource_a(
            traj["resource_A_share"][0],
            lags=args.acf_lags,
            output_path=plots_dir / "resource_A_acf.png",
            show=args.show_plots,
        )

    if not args.no_metric_plot:
        plot_metric_trajectories(
            traj,
            condition_tags=args.condition_tags,
            wsize=args.metric_wsize,
            end=args.end,
            output_path=plots_dir / "metric_trajectories.png",
            show=args.show_plots,
        )

    if not args.no_phase:
        phase_indices = (
            list(range(len(selected)))
            if args.phase_run_index is None
            else list(args.phase_run_index)
        )
        for idx in phase_indices:
            if idx < 0 or idx >= len(selected):
                raise IndexError(
                    f"--phase-run-index {idx} out of range "
                    f"for {len(selected)} selected runs"
                )
        last_n = (
            None
            if args.phase_last_n is None or args.phase_last_n <= 0
            else int(args.phase_last_n)
        )
        shared_limits = compute_shared_phase_limits(
            traj,
            phase_indices,
            y_key=args.phase_y,
            windows=args.phase_windows,
            end=args.end,
            step=args.phase_step,
            last_n=last_n,
        )
        for w, lims in shared_limits.items():
            print(
                f"Shared phase limits (roll={w}): "
                f"s={lims['xlim']}, x={lims['ylim']}, t={lims['zlim']}"
            )
        for idx in phase_indices:
            print(f"Phase portraits for [{idx}] {args.condition_tags[idx]}")
            plot_phase_portraits(
                traj,
                run_idx=idx,
                condition_tag=args.condition_tags[idx],
                y_key=args.phase_y,
                windows=args.phase_windows,
                end=args.end,
                step=args.phase_step,
                arrow_every=args.phase_arrow_every,
                output_dir=plots_dir,
                show=args.show_plots,
                make_2d=bool(args.phase_2d),
                last_n=last_n,
                shared_limits=shared_limits,
            )
        if len(phase_indices) >= 2:
            print(
                "Phase comparison panels: "
                f"{args.condition_tags[phase_indices[0]]} | "
                f"{args.condition_tags[phase_indices[1]]}"
            )
            plot_phase_comparison(
                traj,
                run_indices=phase_indices,
                condition_tags=args.condition_tags,
                y_key=args.phase_y,
                windows=args.phase_windows,
                end=args.end,
                step=args.phase_step,
                arrow_every=args.phase_arrow_every,
                output_dir=plots_dir,
                show=args.show_plots,
                make_2d=bool(args.phase_2d),
                last_n=last_n,
                shared_limits=shared_limits,
            )

    print(f"\nWrote figures under {plots_dir}")


if __name__ == "__main__":
    main()
