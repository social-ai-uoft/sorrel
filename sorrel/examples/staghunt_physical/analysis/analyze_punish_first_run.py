#!/usr/bin/env python3
"""Summarize and plot punish_first probe CSVs for a single training run.

Loads every ``unit_test/punish_first*.csv`` (one row per file), then writes PNGs
under ``analysis/`` by default:

- ``<run>_punish_first_outgroup_by_scenario.png`` — outgroup vs epoch, faceted by layout
- ``<run>_punish_first_hit_rate_and_turn.png`` — pooled hit rate and mean first-punish turn
- ``<run>_punish_first_mean_dist_out_minus_in.png`` — (if column present) mean
  ``dist_outgroup - dist_ingroup`` per probe step, averaged over the episode, vs epoch

Usage::

    python analyze_punish_first_run.py --input /path/to/run_dir
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _analysis_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_unit_test(run_root: Path) -> Path:
    run_root = run_root.resolve()
    if run_root.name == "unit_test":
        return run_root
    ut = run_root / "unit_test"
    if not ut.is_dir():
        raise FileNotFoundError(f"No unit_test/ under {run_root}")
    return ut


def _scenario_label(row: pd.Series) -> str:
    if "partner_kind" in row.index and pd.notna(row.get("partner_kind")) and str(row["partner_kind"]):
        return str(row["partner_kind"])
    uk = row.get("upper_fake_agent_kind")
    lk = row.get("lower_fake_agent_kind")
    if pd.notna(uk) or pd.notna(lk):
        return f"{uk}__{lk}"
    return "unknown"


def load_punish_first_run(unit_test: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for csv_path in sorted(unit_test.glob("punish_first*.csv")):
        try:
            df_one = pd.read_csv(csv_path, nrows=1)
        except pd.errors.EmptyDataError:
            continue
        if df_one.empty:
            continue
        row = df_one.iloc[0].to_dict()
        row["_csv_path"] = str(csv_path)
        rows.append(row)
    if not rows:
        raise ValueError(f"No punish_first CSV rows under {unit_test}")
    out = pd.DataFrame(rows)
    out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce")
    out["_scenario"] = out.apply(_scenario_label, axis=1)
    out["_outgroup"] = pd.to_numeric(out.get("first_punished_is_outgroup"), errors="coerce")
    slot = out["first_punished_fake_slot"].astype(str)
    out["_hit"] = slot.isin(["upper", "lower"])
    out["_turn"] = pd.to_numeric(out["turn_of_first_punishment"], errors="coerce")
    out["_y_plot"] = out["_outgroup"].where(out["_outgroup"].notna(), 0.5)
    if "mean_dist_outgroup_minus_ingroup" in out.columns:
        out["_mean_d_out_minus_in"] = pd.to_numeric(out["mean_dist_outgroup_minus_ingroup"], errors="coerce")
    else:
        out["_mean_d_out_minus_in"] = np.nan
    return out


def _run_stem(unit_test: Path) -> str:
    return unit_test.parent.name if unit_test.name == "unit_test" else unit_test.name


def plot_outgroup_by_scenario(df: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    scenarios = sorted(df["_scenario"].unique())
    n = max(1, len(scenarios))
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False, facecolor="white")
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for si, scen in enumerate(scenarios):
        ax = axes_flat[si]
        ax.set_facecolor("white")
        sub = df[df["_scenario"] == scen].sort_values(["agent_id", "epoch"])
        agents = sorted(sub["agent_id"].unique())
        cmap = plt.cm.tab10(np.linspace(0, 1, max(1, len(agents))))
        for j, aid in enumerate(agents):
            g = sub[sub["agent_id"] == aid].sort_values("epoch")
            y = g["_outgroup"]
            mask = y.notna()
            color = cmap[j % len(cmap)]
            if mask.any():
                ax.plot(g.loc[mask, "epoch"], g.loc[mask, "_outgroup"], color=color, lw=2, label=f"agent {aid}")
            is_na = y.isna().to_numpy()
            yn = y.to_numpy()
            c = np.where(is_na, "#888888", np.where(yn == 1, "#c0392b", "#2e7d32"))
            ax.scatter(g["epoch"], g["_y_plot"], c=c, s=28, edgecolors="k", linewidths=0.4, zorder=3)
        ax.set_title(f"{title_prefix}\n{scen}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("first_punished_is_outgroup")
        ax.set_ylim(-0.12, 1.12)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["0 ingroup", "missing", "1 outgroup"])
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.legend(loc="best", fontsize=7, framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_mean_dist_out_minus_in(df: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    """Mean (over probe steps) of Manhattan d(focal,outgroup) - d(focal,ingroup), vs epoch."""
    if "_mean_d_out_minus_in" not in df.columns or df["_mean_d_out_minus_in"].notna().sum() == 0:
        return
    scenarios = sorted(df["_scenario"].unique())
    n = max(1, len(scenarios))
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False, facecolor="white")
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for si, scen in enumerate(scenarios):
        ax = axes_flat[si]
        ax.set_facecolor("white")
        sub = df[df["_scenario"] == scen].sort_values(["agent_id", "epoch"])
        agents = sorted(sub["agent_id"].unique())
        cmap = plt.cm.tab10(np.linspace(0, 1, max(1, len(agents))))
        for j, aid in enumerate(agents):
            g = sub[sub["agent_id"] == aid].sort_values("epoch")
            y = g["_mean_d_out_minus_in"]
            mask = y.notna()
            if mask.any():
                ax.plot(
                    g.loc[mask, "epoch"],
                    g.loc[mask, "_mean_d_out_minus_in"],
                    color=cmap[j % len(cmap)],
                    lw=2,
                    marker="o",
                    ms=4,
                    label=f"agent {aid}",
                )
        ax.axhline(0.0, color="gray", lw=1.0, linestyle=":")
        ax.set_title(f"{title_prefix}\n{scen}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mean d(outgroup) − d(ingroup) over probe steps")
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.legend(loc="best", fontsize=7, framealpha=0.95)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_mean_dist_out_minus_in_over_time(df: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    """Aggregate mean(d_out - d_in) by epoch with error bars (mean ± SEM).

    SEM is computed across agent means within each epoch (per scenario).
    """
    if "_mean_d_out_minus_in" not in df.columns or df["_mean_d_out_minus_in"].notna().sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.set_facecolor("white")

    scenarios = sorted(df["_scenario"].unique())
    for scen in scenarios:
        sub = df[(df["_scenario"] == scen) & (df["_mean_d_out_minus_in"].notna())].copy()
        if sub.empty:
            continue

        # First average within (epoch, agent) to avoid overweighting any duplicated rows.
        per_agent = (
            sub.groupby(["epoch", "agent_id"], sort=True)["_mean_d_out_minus_in"]
            .mean()
            .reset_index()
        )
        g = per_agent.groupby("epoch", sort=True)["_mean_d_out_minus_in"]
        mean = g.mean()
        n = g.count()
        std = g.std(ddof=1)
        sem = std / np.sqrt(n.clip(lower=1))

        x = mean.index.to_numpy()
        ax.errorbar(
            x,
            mean.to_numpy(),
            yerr=sem.to_numpy(),
            marker="o",
            ms=4,
            lw=2,
            capsize=3,
            label=f"{scen} (mean±SEM across agents)",
        )

    ax.axhline(0.0, color="gray", lw=1.0, linestyle=":")
    ax.set_title(f"{title_prefix} — mean(d_out − d_in) across time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mean(d_out − d_in) over probe steps")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(loc="best", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_hit_rate_and_turn(df: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), facecolor="white")
    for ax in (ax0, ax1):
        ax.set_facecolor("white")

    scenarios = sorted(df["_scenario"].unique())
    for scen in scenarios:
        sub = df[df["_scenario"] == scen]
        g = sub.groupby("epoch", sort=True).agg(hit_mean=("_hit", "mean"), turn_hit=("_turn", "mean")).reset_index()
        ax0.plot(g["epoch"], g["hit_mean"], marker="o", ms=4, label=scen, lw=2)
        hits = sub[sub["_hit"]].groupby("epoch", sort=True)["_turn"].mean().reset_index()
        if not hits.empty:
            ax1.plot(hits["epoch"], hits["_turn"], marker="o", ms=4, label=scen, lw=2)

    ax0.set_ylabel("P(any fake punished)")
    ax0.set_xlabel("Epoch")
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_title(f"{title_prefix} — punish hit rate (mean over agents)")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.35, linestyle="--")

    ax1.set_ylabel("Mean turn_of_first_punishment (hits only)")
    ax1.set_xlabel("Epoch")
    ax1.set_title(f"{title_prefix} — latency of first punish when hit")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.35, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _collapse_target_kind_y(ys: pd.Series) -> float:
    """Collapse multiple probe scenarios at the same epoch to one y in {0, 0.5, 1}.

    Encoding per row: ingroup=0, no punish=0.5, outgroup=1.
    Rule: if any scenario is outgroup -> 1; elif any ingroup -> 0; else 0.5.
    """
    arr = pd.to_numeric(ys, errors="coerce").fillna(0.5).to_numpy(dtype=float)
    if np.any(arr == 1.0):
        return 1.0
    if np.any(arr == 0.0):
        return 0.0
    return 0.5


def plot_slot_counts(df: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    ax.set_facecolor("white")
    counts = df["first_punished_fake_slot"].astype(str).value_counts()
    order = ["upper", "lower", "none", "unknown"]
    idx = [k for k in order if k in counts.index]
    rest = [k for k in counts.index if k not in idx]
    idx = idx + rest
    vals = [counts[k] for k in idx]
    ax.bar(idx, vals, color=["#3498db", "#9b59b6", "#95a5a6", "#e74c3c"][: len(idx)])
    ax.set_ylabel("Count (all CSV rows)")
    ax.set_title(f"{title_prefix} — first_punished_fake_slot")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_per_agent_target_kind(df: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    """One subplot per focal agent: first-punish target kind over epochs (single line).

    Y encoding: ingroup = 0, no punish = 0.5, outgroup = 1.
    Both probe layouts at the same epoch are collapsed to one value (see ``_collapse_target_kind_y``).
    """
    agents = sorted(df["agent_id"].dropna().unique())
    n = len(agents)
    ncols = 3 if n >= 3 else max(1, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.8 * nrows), squeeze=False, facecolor="white")
    axes_flat = axes.ravel()

    for ax in axes_flat:
        ax.set_facecolor("white")
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for i, aid in enumerate(agents):
        ax = axes_flat[i]
        sub = df[df["agent_id"] == aid].copy()
        sub["_y_kind"] = pd.to_numeric(sub["_outgroup"], errors="coerce").fillna(0.5)
        g = (
            sub.groupby("epoch", sort=True)["_y_kind"]
            .agg(_collapse_target_kind_y)
            .reset_index()
            .sort_values("epoch")
        )
        if g.empty:
            continue
        ax.plot(
            g["epoch"],
            g["_y_kind"],
            color="C0",
            linewidth=2.0,
            marker="o",
            markersize=4,
            alpha=0.92,
        )

        ax.set_title(f"agent {aid}")
        ax.set_xlabel("Epoch")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["ingroup", "no punish", "outgroup"])
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(
        f"{title_prefix} — per-agent first punish target kind (scenarios collapsed per epoch)",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="Training run directory (contains unit_test/).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNGs (default: this script's analysis/).",
    )
    args = p.parse_args()

    ut = _resolve_unit_test(args.input)
    out_dir = args.output_dir.resolve() if args.output_dir is not None else _analysis_dir()
    stem = re.sub(r"[^\w\-.]+", "_", _run_stem(ut))[:180]
    title_prefix = _run_stem(ut)

    df = load_punish_first_run(ut)

    n = len(df)
    hit = float(df["_hit"].mean())
    og_hit = df.loc[df["_hit"], "_outgroup"].mean()
    print(f"Loaded {n} punish_first rows from {ut}")
    print(f"Punish hit rate (any upper/lower first): {hit:.3f}")
    if df["_hit"].any():
        print(f"Mean first_punished_is_outgroup when hit: {og_hit:.3f}")
        print(f"Mean turn_of_first_punishment when hit: {df.loc[df['_hit'], '_turn'].mean():.2f}")
    print("Slot counts:\n", df["first_punished_fake_slot"].astype(str).value_counts())

    if df["_mean_d_out_minus_in"].notna().any():
        md = df["_mean_d_out_minus_in"].dropna()
        print(
            f"Mean distance bias (mean over probe steps): "
            f"global mean of mean(d_out−d_in) = {md.mean():.4f} "
            f"(n rows with value {len(md)})"
        )
        for scen in sorted(df["_scenario"].unique()):
            m = df.loc[df["_scenario"] == scen, "_mean_d_out_minus_in"].dropna()
            if len(m):
                print(f"  {scen}: mean = {m.mean():.4f} (n={len(m)})")

    plot_outgroup_by_scenario(df, out_dir / f"{stem}_punish_first_outgroup_by_scenario.png", title_prefix)
    plot_hit_rate_and_turn(df, out_dir / f"{stem}_punish_first_hit_rate_and_turn.png", title_prefix)
    plot_slot_counts(df, out_dir / f"{stem}_punish_first_slot_counts.png", title_prefix)
    plot_per_agent_target_kind(df, out_dir / f"{stem}_punish_first_per_agent_target_kind.png", title_prefix)

    dist_png = out_dir / f"{stem}_punish_first_mean_dist_out_minus_in.png"
    plot_mean_dist_out_minus_in(df, dist_png, title_prefix)
    dist_time_png = out_dir / f"{stem}_punish_first_mean_dist_out_minus_in_over_time.png"
    plot_mean_dist_out_minus_in_over_time(df, dist_time_png, title_prefix)

    print(f"Wrote PNGs under {out_dir}:")
    names = [
        f"{stem}_punish_first_outgroup_by_scenario.png",
        f"{stem}_punish_first_hit_rate_and_turn.png",
        f"{stem}_punish_first_slot_counts.png",
        f"{stem}_punish_first_per_agent_target_kind.png",
    ]
    if dist_png.exists():
        names.append(f"{stem}_punish_first_mean_dist_out_minus_in.png")
    if dist_time_png.exists():
        names.append(f"{stem}_punish_first_mean_dist_out_minus_in_over_time.png")
    for name in names:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
