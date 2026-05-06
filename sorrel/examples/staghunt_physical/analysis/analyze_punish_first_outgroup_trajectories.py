#!/usr/bin/env python3
"""Plot punish-first probe outgroup trajectories over training epochs.

The probe CSV column ``first_punished_is_outgroup`` is 1 if the first punished
fake's ``agent_kind`` differs from the focal's (outgroup), 0 if same (ingroup),
and empty if there was no / unknown first punishment. This matches the idea of
``punish_outgroup_or_not`` as a binary outcome when a punish occurred.

The default figure uses one subplot per focal agent (or per run+agent when
``--recursive`` loads multiple runs), with all probe scenarios **collapsed** at each
epoch: if any scenario has outgroup (1) then y=1; elif any has ingroup (0) then y=0;
else y=0.5 (no punish / unknown in all scenarios).

Typical layout::

    <run_dir>/unit_test/punish_first_epoch_<epoch>_agent_<id>_map_<map>_partner_<...>.csv

Examples::

    # Writes sorrel/examples/staghunt_physical/analysis/punish_first_outgroup_<run_name>.png
    python analyze_punish_first_outgroup_trajectories.py \\
        --input /path/to/run_or_unit_test_dir

    # Custom output path (any directory)
    python analyze_punish_first_outgroup_trajectories.py \\
        --input /path/to/run_or_unit_test_dir \\
        --output /tmp/punish_first_outgroup.png

    # All runs under a parent folder (each containing unit_test/)
    python analyze_punish_first_outgroup_trajectories.py \\
        --input /path/to/data/runs_tageffect_v3 \\
        --recursive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMN = "first_punished_is_outgroup"
EPOCH_COLUMN = "epoch"


def _analysis_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_output_png(input_path: Path, unit_test_dirs: list[Path], recursive: bool) -> Path:
    """Save next to this script under analysis/ with a name derived from the input run(s)."""
    if len(unit_test_dirs) == 1 and unit_test_dirs[0].name == "unit_test":
        stem = unit_test_dirs[0].parent.name
    elif len(unit_test_dirs) == 1:
        stem = unit_test_dirs[0].name
    else:
        stem = f"{input_path.resolve().name}_{len(unit_test_dirs)}runs"
    if recursive and len(unit_test_dirs) > 1:
        stem = f"{input_path.resolve().name}_recursive_{len(unit_test_dirs)}runs"
    return _analysis_dir() / f"punish_first_outgroup_{stem}.png"


def _resolve_unit_test_dirs(root: Path, recursive: bool) -> list[Path]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        raise ValueError("--input must be a directory")
    if root.name == "unit_test":
        return [root]
    direct = root / "unit_test"
    if direct.is_dir():
        return [direct]
    if not recursive:
        raise FileNotFoundError(
            f"No unit_test/ under {root}. Pass a run directory, a unit_test directory, "
            "or add --recursive to search descendants."
        )
    found = sorted({p.parent for p in root.rglob("unit_test/punish_first*.csv")})
    if not found:
        raise FileNotFoundError(
            f"No unit_test/punish_first*.csv under {root} (recursive search)."
        )
    return found


def _load_punish_first_csvs(unit_test_dirs: list[Path], run_tag: str | None) -> pd.DataFrame:
    rows: list[dict] = []
    for ut in unit_test_dirs:
        tag = run_tag
        if tag is None:
            tag = ut.parent.name if ut.name == "unit_test" else ut.name
        for csv_path in sorted(ut.glob("punish_first*.csv")):
            try:
                df_one = pd.read_csv(csv_path, nrows=1)
            except pd.errors.EmptyDataError:
                continue
            if df_one.empty:
                continue
            if REQUIRED_COLUMN not in df_one.columns:
                print(f"Skip (missing {REQUIRED_COLUMN}): {csv_path}", file=sys.stderr)
                continue
            row = df_one.iloc[0].to_dict()
            row["_csv_path"] = str(csv_path)
            row["_run_tag"] = tag
            rows.append(row)
    if not rows:
        raise ValueError("No non-empty punish_first CSV rows loaded.")
    return pd.DataFrame(rows)


def _scenario_key(row: pd.Series) -> str:
    if "partner_kind" in row.index and pd.notna(row["partner_kind"]) and str(row["partner_kind"]):
        return str(row["partner_kind"])
    uk = row.get("upper_fake_agent_kind")
    lk = row.get("lower_fake_agent_kind")
    if pd.notna(uk) or pd.notna(lk):
        return f"{uk}__{lk}"
    return "unknown"


def _prepare_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    if EPOCH_COLUMN not in df.columns:
        raise ValueError(f"Missing column {EPOCH_COLUMN!r} in CSVs.")
    out = df.copy()
    out[EPOCH_COLUMN] = pd.to_numeric(out[EPOCH_COLUMN], errors="coerce")
    out["_outgroup"] = pd.to_numeric(out[REQUIRED_COLUMN], errors="coerce")
    out["_scenario"] = out.apply(_scenario_key, axis=1)
    out["_series"] = (
        out["_run_tag"].astype(str)
        + " | agent "
        + out["agent_id"].astype(str)
        + " | "
        + out["_scenario"].astype(str)
    )
    # y position for scatter: NaN (no punish / unknown) -> mid band so points are visible
    out["_y_plot"] = out["_outgroup"].where(out["_outgroup"].notna(), 0.5)
    return out


def _collapse_outgroup_across_scenarios(ys: pd.Series) -> float:
    """Collapse multiple scenarios at the same (run, agent, epoch) to one y in {0, 0.5, 1}.

    Same rule as ``analyze_punish_first_run._collapse_target_kind_y``: any outgroup
    (1) -> 1; elif any ingroup (0) -> 0; else 0.5 (all missing / no punish).
    """
    arr = pd.to_numeric(ys, errors="coerce").fillna(0.5).to_numpy(dtype=float)
    if np.any(arr == 1.0):
        return 1.0
    if np.any(arr == 0.0):
        return 0.0
    return 0.5


def _collapsed_per_agent_epoch(plot_df: pd.DataFrame) -> pd.DataFrame:
    g = (
        plot_df.groupby(["_run_tag", "agent_id", EPOCH_COLUMN], sort=True)["_outgroup"]
        .agg(_collapse_outgroup_across_scenarios)
        .reset_index(name="_y")
    )
    g["_label"] = (
        g["_run_tag"].astype(str) + " | agent " + g["agent_id"].astype(str)
        if g["_run_tag"].nunique() > 1
        else "agent " + g["agent_id"].astype(str)
    )
    return g


def _plot_collapsed_per_agent_subplots(collapsed: pd.DataFrame, title: str, output_path: Path) -> None:
    """One subplot per agent (or run|agent), scenarios already collapsed per epoch."""
    series_keys = sorted(collapsed["_label"].unique())
    n = len(series_keys)
    ncols = 3 if n >= 3 else max(1, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.9 * nrows), squeeze=False, facecolor="white")
    axes_flat = axes.ravel()

    for ax in axes_flat:
        ax.set_facecolor("white")
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    cmap = plt.cm.tab10(np.linspace(0, 1, max(1, n)))
    for i, lab in enumerate(series_keys):
        ax = axes_flat[i]
        sub = collapsed[collapsed["_label"] == lab].sort_values(EPOCH_COLUMN)
        y = sub["_y"]
        color = cmap[i % len(cmap)]
        ax.plot(sub[EPOCH_COLUMN], y, color=color, lw=2.2, marker="o", ms=3.8, zorder=2)
        c = np.where(y == 0.5, "#888888", np.where(y == 1.0, "#c0392b", "#2e7d32"))
        ax.scatter(
            sub[EPOCH_COLUMN],
            y,
            c=c,
            s=26,
            edgecolors="k",
            linewidths=0.35,
            zorder=3,
        )
        ax.set_title(str(lab), fontsize=10)
        ax.set_xlabel("Training epoch (probe)")
        ax.set_ylim(-0.12, 1.12)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["0 ingroup", "missing", "1 outgroup"])
        ax.grid(True, alpha=0.35, linestyle="--")

    # Add y-label only on the left column for readability.
    for row_idx in range(nrows):
        axes[row_idx, 0].set_ylabel("first_punished_is_outgroup (collapsed)")

    fig.suptitle(title, y=1.01, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Run directory (contains unit_test/), a unit_test/ folder, or a parent of runs with --recursive.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="If set, find every .../unit_test/ under --input that has punish_first*.csv.",
    )
    p.add_argument(
        "--run-tag",
        default=None,
        help="Label for run (default: parent folder name of unit_test).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. If omitted, writes under this script's analysis/ folder as "
            "punish_first_outgroup_<run_name>.png (see module docstring)."
        ),
    )
    p.add_argument(
        "--title",
        default="first_punished_is_outgroup vs epoch (punish_first probe)",
        help="Figure title.",
    )
    args = p.parse_args()

    unit_dirs = _resolve_unit_test_dirs(args.input, args.recursive)
    output_path = args.output if args.output is not None else _default_output_png(
        args.input, unit_dirs, args.recursive
    )

    df = _load_punish_first_csvs(unit_dirs, args.run_tag)
    plot_df = _prepare_plot_df(df)

    n_valid = plot_df["_outgroup"].notna().sum()
    n_total = len(plot_df)
    if n_valid == 0:
        print(
            f"Note: all {n_total} rows have empty/NaN {REQUIRED_COLUMN!r} "
            "(e.g. no fake punished in probe). Plot uses y=0.5 for those points in gray.",
            file=sys.stderr,
        )
    elif n_valid < n_total * 0.05:
        print(
            f"Note: only {n_valid}/{n_total} rows have a non-empty {REQUIRED_COLUMN!r}; "
            "missing values are shown at y=0.5 in gray.",
            file=sys.stderr,
        )

    collapsed = _collapsed_per_agent_epoch(plot_df)
    _plot_collapsed_per_agent_subplots(collapsed, args.title, output_path)
    print(
        f"Wrote {output_path.resolve()} ({len(collapsed)} collapsed points from "
        f"{len(plot_df)} raw rows, {collapsed['_label'].nunique()} lines) — "
        "scenarios collapsed per epoch."
    )


if __name__ == "__main__":
    main()
