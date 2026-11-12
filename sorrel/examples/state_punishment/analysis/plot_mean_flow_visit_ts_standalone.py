#!/usr/bin/env python3
"""Standalone mean-flow + per-cell visit time-series plotter.

Reproduces the style of:
  mean_flow_observable_pair_smooth40_visit_ts_15x15_last100k.png

Input data: one or more trajectories of (time, x) and (time, s).
Pass them as arrays, DataFrames with columns t/x/s, or CSV paths.

Example (Python API)::

    import numpy as np
    from plot_mean_flow_visit_ts_standalone import plot_mean_flow_visit_ts

    t = np.arange(100_000)
    x = ...  # shape (T,)
    s = ...  # shape (T,)

    plot_mean_flow_visit_ts(
        [{"t": t, "x": x, "s": s}],          # one condition, one trajectory
        "out.png",
        labels=["my condition"],
        n_bins=15,
        smooth_window=40,
        last_n_steps=100_000,
        n_time_bins=5000,
        x_lim=(0.0, 1.0),
        s_lim=(-0.5, 9.5),
    )

    # Two side-by-side panels, each with several seed trajectories:
    plot_mean_flow_visit_ts(
        {
            "imperfect": [{"t": t, "x": x1, "s": s1}, {"t": t, "x": x2, "s": s2}],
            "perfect":   [{"t": t, "x": xp, "s": sp}],
        },
        "pair.png",
    )

Example (CLI)::

    python plot_mean_flow_visit_ts_standalone.py \\
        --csv run_a.csv --csv run_b.csv \\
        --out mean_flow_visit_ts.png \\
        --n-bins 15 --smooth 40 --last-n 100000 --n-time-bins 5000

CSV must have columns: t (or time), x, s.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def _as_frame(traj: Any) -> pd.DataFrame:
    """Normalize one trajectory into a DataFrame with columns t, x, s."""
    if isinstance(traj, pd.DataFrame):
        df = traj.copy()
    elif isinstance(traj, Mapping):
        df = pd.DataFrame(traj)
    else:
        raise TypeError(
            "trajectory must be a DataFrame or mapping with keys t/x/s; "
            f"got {type(traj)!r}"
        )

    rename = {}
    if "t" not in df.columns and "time" in df.columns:
        rename["time"] = "t"
    if rename:
        df = df.rename(columns=rename)

    missing = [c for c in ("t", "x", "s") if c not in df.columns]
    if missing:
        raise ValueError(f"trajectory missing columns {missing}; have {list(df.columns)}")

    out = df[["t", "x", "s"]].copy()
    out = out.sort_values("t").reset_index(drop=True)
    out["t"] = out["t"].astype(float)
    out["x"] = out["x"].astype(float)
    out["s"] = out["s"].astype(float)
    out["delta_s"] = out["s"].diff().fillna(0.0)
    out["dx"] = out["x"].diff().fillna(0.0)
    return out


def _normalize_panel_input(
    data: Any,
) -> dict[str, list[pd.DataFrame]]:
    """Accept flexible input; return {panel_label: [frames...]}."""
    # Single trajectory
    if isinstance(data, (pd.DataFrame, Mapping)) and not (
        isinstance(data, Mapping)
        and data
        and all(isinstance(v, (list, tuple)) for v in data.values())
    ):
        # Mapping of arrays (t,x,s) vs mapping of panels
        if isinstance(data, Mapping) and set(data.keys()) >= {"x", "s"}:
            return {"series": [_as_frame(data)]}
        if isinstance(data, pd.DataFrame):
            return {"series": [_as_frame(data)]}

    # List of trajectories -> one panel
    if isinstance(data, (list, tuple)):
        return {"series": [_as_frame(tr) for tr in data]}

    # Dict of panels
    if isinstance(data, Mapping):
        panels: dict[str, list[pd.DataFrame]] = {}
        for label, value in data.items():
            if isinstance(value, (list, tuple)):
                panels[str(label)] = [_as_frame(tr) for tr in value]
            else:
                panels[str(label)] = [_as_frame(value)]
        return panels

    raise TypeError(
        "data must be a trajectory, a list of trajectories, or a dict of "
        f"panel_label -> trajectory/list; got {type(data)!r}"
    )


def _tail_df(df: pd.DataFrame, last_n_steps: int | None) -> pd.DataFrame:
    if last_n_steps is None or last_n_steps <= 0 or len(df) <= last_n_steps:
        return df
    return df.iloc[-int(last_n_steps) :].copy()


def _prepare_flow_frame(
    df: pd.DataFrame,
    *,
    last_n_steps: int | None,
    smooth_window: int,
    smooth: bool,
) -> pd.DataFrame:
    out = _tail_df(df, last_n_steps).copy()
    if smooth and smooth_window is not None and int(smooth_window) > 1:
        w = int(smooth_window)
        out["s"] = (
            pd.Series(out["s"].to_numpy(dtype=float))
            .rolling(w, min_periods=1, center=True)
            .mean()
            .to_numpy(dtype=float)
        )
        out["x"] = (
            pd.Series(out["x"].to_numpy(dtype=float))
            .rolling(w, min_periods=1, center=True)
            .mean()
            .to_numpy(dtype=float)
        )
        out["delta_s"] = out["s"].diff().fillna(0.0)
        out["dx"] = out["x"].diff().fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Field / mosaic computation
# ---------------------------------------------------------------------------

def _mean_flow_field(
    frames: list[pd.DataFrame],
    *,
    n_x_bins: int,
    n_s_bins: int,
    x_min: float,
    x_max: float,
    s_min: float,
    s_max: float,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (S, X, U=mean Δs, V=mean Δx, counts, s_edges, x_edges)."""
    if not frames:
        empty = np.array([])
        return empty, empty, empty, empty, empty, empty, empty

    s_edges = np.linspace(float(s_min), float(s_max), int(n_s_bins) + 1)
    x_edges = np.linspace(float(x_min), float(x_max), int(n_x_bins) + 1)
    n_s = len(s_edges) - 1
    n_x = len(x_edges) - 1
    sum_u = np.zeros((n_x, n_s))
    sum_v = np.zeros((n_x, n_s))
    counts = np.zeros((n_x, n_s))
    s_lo, s_hi = float(s_edges[0]), float(s_edges[-1])

    for df in frames:
        s = df["s"].to_numpy(dtype=float)
        x = df["x"].to_numpy(dtype=float)
        u = df["delta_s"].to_numpy(dtype=float)
        v = df["dx"].to_numpy(dtype=float)
        mask = (
            np.isfinite(s)
            & np.isfinite(x)
            & np.isfinite(u)
            & np.isfinite(v)
            & (x >= float(x_min))
            & (x <= float(x_max))
            & (s >= s_lo)
            & (s <= s_hi)
        )
        s, x, u, v = s[mask], x[mask], u[mask], v[mask]
        if len(s) == 0:
            continue
        si = np.clip(np.digitize(s, s_edges) - 1, 0, n_s - 1)
        xi = np.clip(np.digitize(x, x_edges) - 1, 0, n_x - 1)
        np.add.at(sum_u, (xi, si), u)
        np.add.at(sum_v, (xi, si), v)
        np.add.at(counts, (xi, si), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        U = np.where(counts >= min_count, sum_u / counts, np.nan)
        V = np.where(counts >= min_count, sum_v / counts, np.nan)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    S, X = np.meshgrid(s_centers, x_centers)
    return S, X, U, V, counts, s_edges, x_edges


def _cell_visit_timeseries(
    frames: list[pd.DataFrame],
    *,
    s_edges: np.ndarray,
    x_edges: np.ndarray,
    n_time_bins: int,
) -> np.ndarray:
    """Pooled visit fraction vs coarsened time; shape (n_x, n_s, K)."""
    s_edges = np.asarray(s_edges, dtype=float)
    x_edges = np.asarray(x_edges, dtype=float)
    n_s = len(s_edges) - 1
    n_x = len(x_edges) - 1
    k = max(int(n_time_bins), 1)
    if n_s <= 0 or n_x <= 0 or not frames:
        return np.zeros((max(n_x, 0), max(n_s, 0), k), dtype=float)

    lengths = [
        int(min(len(df["s"]), len(df["x"])))
        for df in frames
        if "s" in df.columns and "x" in df.columns
    ]
    lengths = [n for n in lengths if n >= 1]
    if not lengths:
        return np.zeros((n_x, n_s, k), dtype=float)

    n_steps = int(min(lengths))
    counts = np.zeros((n_x, n_s, k), dtype=float)
    n_seeds = 0
    x_lo = float(x_edges[0])
    x_hi = float(x_edges[-1])
    t = np.arange(n_steps, dtype=int)
    ti = np.minimum((t * k) // n_steps, k - 1)

    for df in frames:
        s = df["s"].to_numpy(dtype=float)[:n_steps]
        x = df["x"].to_numpy(dtype=float)[:n_steps]
        mask = np.isfinite(s) & np.isfinite(x) & (x >= x_lo) & (x <= x_hi)
        n_seeds += 1
        if not np.any(mask):
            continue
        si = np.clip(np.digitize(s[mask], s_edges) - 1, 0, n_s - 1)
        xi = np.clip(np.digitize(x[mask], x_edges) - 1, 0, n_x - 1)
        np.add.at(counts, (xi, si, ti[mask]), 1.0)

    steps_per_block = np.bincount(ti, minlength=k).astype(float)
    steps_per_block = np.maximum(steps_per_block, 1.0)
    denom = steps_per_block[None, None, :] * float(max(n_seeds, 1))
    return counts / denom


def _visit_mosaic_rgb(
    f: np.ndarray,
    *,
    line_color: str,
    ymax: float,
    patch_w: int = 96,
    patch_h: int = 96,
) -> np.ndarray:
    """Rasterize per-cell visit series into an RGB mosaic (origin=lower)."""
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError(f"expected f with shape (n_x, n_s, K), got {f.shape}")
    n_x, n_s, k = f.shape
    img = np.ones((n_s * patch_h, n_x * patch_w, 3), dtype=float)
    rgba = mpl.colors.to_rgba(line_color)
    line_rgb = np.array(rgba[:3], dtype=float)
    y_top = max(float(ymax), 1e-12)

    tot = np.nansum(f, axis=-1)
    pos = tot[np.isfinite(tot) & (tot > 0)]
    tot_ref = float(np.median(pos)) if len(pos) else 1.0
    tot_ref = max(tot_ref, 1e-12)
    g_lo, g_hi = 0.50, 1.00

    def _gray_rgb(visits: float) -> np.ndarray:
        frac = float(visits) / (float(visits) + tot_ref)
        g = float(np.clip(g_lo + (g_hi - g_lo) * frac, g_lo, g_hi))
        return np.array([g, g, g], dtype=float)

    for xi in range(n_x):
        for si in range(n_s):
            series = f[xi, si]
            y0 = si * patch_h
            x0 = xi * patch_w
            patch = img[y0 : y0 + patch_h, x0 : x0 + patch_w]
            visits = float(np.nansum(series)) if np.isfinite(series).any() else 0.0
            face_rgb = _gray_rgb(max(visits, 0.0))
            if visits <= 0.0:
                patch[:] = face_rgb
                patch[0, :] = 0.55
                patch[-1, :] = 0.55
                patch[:, 0] = 0.55
                patch[:, -1] = 0.55
                continue

            fig = Figure(figsize=(patch_w / 100.0, patch_h / 100.0), dpi=100)
            canvas = FigureCanvasAgg(fig)
            fig.patch.set_facecolor(tuple(face_rgb))
            ax = fig.add_axes([0.1, 0.14, 0.8, 0.72])
            tt = np.arange(k, dtype=float)
            ax.fill_between(tt, series, color=line_rgb, alpha=0.25, linewidth=0)
            ax.plot(tt, series, color=line_rgb, lw=1.1)
            ax.set_xlim(0.0, max(k - 1, 1))
            ax.set_ylim(0.0, y_top)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(tuple(face_rgb))
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("0.65")
            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba(), dtype=float)[:, :, :3] / 255.0
            plt.close(fig)
            buf = np.flipud(buf)
            bh, bw = buf.shape[:2]
            if (bh, bw) != (patch_h, patch_w):
                rs = (np.linspace(0, bh - 1, patch_h)).astype(int)
                cs = (np.linspace(0, bw - 1, patch_w)).astype(int)
                buf = buf[rs][:, cs]
            patch[:] = buf
    return img


def _fixed_length_arrows(
    u: np.ndarray,
    v: np.ndarray,
    *,
    s_span: float,
    x_span: float,
    length_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale (u, v) to equal on-screen length (axis-fraction coordinates)."""
    s_span = max(float(s_span), 1e-12)
    x_span = max(float(x_span), 1e-12)
    u_n = u / s_span
    v_n = v / x_span
    speed = np.sqrt(u_n ** 2 + v_n ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_hat = np.where(speed > 0, length_frac * u_n / speed * s_span, 0.0)
        v_hat = np.where(speed > 0, length_frac * v_n / speed * x_span, 0.0)
    return u_hat, v_hat


# ---------------------------------------------------------------------------
# Main plotter
# ---------------------------------------------------------------------------

DEFAULT_LINE_COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
)


def plot_mean_flow_visit_ts(
    data: Any,
    out_path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    line_colors: Sequence[str] | None = None,
    n_bins: int = 15,
    n_x_bins: int | None = None,
    n_s_bins: int | None = None,
    n_x_cells: int | None = None,
    n_s_cells: int | None = None,
    min_count: int = 1,
    arrow_len: float = 0.05,
    x_scale: float = 10.0,
    last_n_steps: int | None = 100_000,
    smooth_window: int = 40,
    n_time_bins: int = 5000,
    x_lim: tuple[float, float] = (0.0, 1.0),
    s_lim: tuple[float, float] = (-0.5, 9.5),
    x_label: str = "$x$",
    s_label: str = "$s$",
    title: str | None = None,
    dpi: int = 200,
    figsize_per_panel: tuple[float, float] = (5.5, 5.6),
) -> Path:
    """Make a mean-flow quiver over a mosaic of per-cell visit time series.

    Parameters
    ----------
    data
        One of:
          - dict with keys ``t``/``time``, ``x``, ``s`` (arrays)
          - DataFrame with those columns
          - list of the above (multiple seeds, one panel)
          - dict ``{panel_label: trajectory_or_list}`` (side-by-side panels)
    out_path
        Output PNG path.
    n_bins
        Convenience: sets both arrow bins and inset cells to n×n when the
        finer ``n_*_bins`` / ``n_*_cells`` args are left as None.
    smooth_window
        Rolling window applied per trajectory before mean-flow arrows.
        Insets always use raw (unsmoothed) visit frequency.
    last_n_steps
        Keep only the last N rows of each trajectory (None = keep all).
    n_time_bins
        Number of coarsened time blocks for the inset visit curves (K).
    x_scale
        Weight on Δx inside the scaled-speed color:
        ``sqrt(Δs² + (x_scale·Δx)²)``.
    """
    panels = _normalize_panel_input(data)
    panel_names = list(panels.keys())
    if labels is not None:
        if len(labels) != len(panel_names):
            raise ValueError(
                f"labels length {len(labels)} != number of panels {len(panel_names)}"
            )
        panel_names = [str(lb) for lb in labels]
        panels = dict(zip(panel_names, panels.values()))

    n_x_bins = int(n_x_bins if n_x_bins is not None else n_bins)
    n_s_bins = int(n_s_bins if n_s_bins is not None else n_bins)
    n_x_cells = int(n_x_cells if n_x_cells is not None else n_bins)
    n_s_cells = int(n_s_cells if n_s_cells is not None else n_bins)

    x_min_bin = float(x_lim[0])
    x_max_bin = float(x_lim[1])
    s_min_bin = float(s_lim[0])
    s_max_bin = float(s_lim[1])

    cell_x_edges = np.linspace(x_min_bin, x_max_bin, n_x_cells + 1)
    cell_s_edges = np.linspace(s_min_bin, s_max_bin, n_s_cells + 1)

    fields: dict[str, tuple] = {}
    visit_f: dict[str, np.ndarray] = {}
    n_seeds_by_panel: dict[str, int] = {}

    for name, frames in panels.items():
        flow_frames = [
            _prepare_flow_frame(
                fr, last_n_steps=last_n_steps, smooth_window=smooth_window, smooth=True
            )
            for fr in frames
        ]
        visit_frames = [
            _prepare_flow_frame(
                fr, last_n_steps=last_n_steps, smooth_window=smooth_window, smooth=False
            )
            for fr in frames
        ]
        fields[name] = _mean_flow_field(
            flow_frames,
            n_x_bins=n_x_bins,
            n_s_bins=n_s_bins,
            x_min=x_min_bin,
            x_max=x_max_bin,
            s_min=s_min_bin,
            s_max=s_max_bin,
            min_count=min_count,
        )
        visit_f[name] = _cell_visit_timeseries(
            visit_frames,
            s_edges=cell_s_edges,
            x_edges=cell_x_edges,
            n_time_bins=n_time_bins,
        )
        n_seeds_by_panel[name] = len(frames)

    # Shared speed scale for arrow colors.
    speeds_all = []
    for S, X, U, V, counts, s_edges, x_edges in fields.values():
        mask = np.isfinite(U) & np.isfinite(V)
        if not np.any(mask):
            continue
        speeds_all.append(np.sqrt(U[mask] ** 2 + (x_scale * V[mask]) ** 2))
    speed_vals = np.concatenate(speeds_all) if speeds_all else np.array([0.0])
    speed_vals = speed_vals[np.isfinite(speed_vals) & (speed_vals > 0)]
    vmin = float(np.percentile(speed_vals, 5)) if len(speed_vals) else 0.0
    vmax = float(np.percentile(speed_vals, 95)) if len(speed_vals) else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("plasma")

    # Shared inset y-scale.
    f_all = np.concatenate([visit_f[n].ravel() for n in panel_names])
    f_pos = f_all[np.isfinite(f_all) & (f_all > 0)]
    y_max = float(np.percentile(f_pos, 95)) if len(f_pos) else 1.0
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 1.0

    colors = list(line_colors) if line_colors is not None else list(DEFAULT_LINE_COLORS)
    while len(colors) < len(panel_names):
        colors.extend(DEFAULT_LINE_COLORS)

    n_panels = len(panel_names)
    fig_w = figsize_per_panel[0] * n_panels + (0.5 if n_panels > 1 else 0.0)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(fig_w, figsize_per_panel[1]),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = np.array([axes])

    s_span = float(s_lim[1] - s_lim[0])
    x_span = float(x_lim[1] - x_lim[0])
    smooth_note = (
        "no pre-smoothing"
        if smooth_window is None or int(smooth_window) <= 1
        else f"per-traj rolling smooth={int(smooth_window)} then pool"
    )
    last_note = (
        "all steps"
        if last_n_steps is None or last_n_steps <= 0
        else f"last {int(last_n_steps):,} steps"
    )

    for ax, name, color in zip(axes, panel_names, colors):
        S, X, U, V, counts, s_edges, x_edges = fields[name]
        f = visit_f[name]
        mosaic = _visit_mosaic_rgb(f, line_color=color, ymax=y_max)
        ax.imshow(
            mosaic,
            origin="lower",
            extent=(
                float(cell_x_edges[0]),
                float(cell_x_edges[-1]),
                float(cell_s_edges[0]),
                float(cell_s_edges[-1]),
            ),
            aspect="auto",
            interpolation="nearest",
            alpha=0.6,
            zorder=0,
        )
        for xv in cell_x_edges:
            ax.axvline(float(xv), color="0.55", lw=0.6, zorder=1)
        for sv in cell_s_edges:
            ax.axhline(float(sv), color="0.55", lw=0.6, zorder=1)

        mask = np.isfinite(U) & np.isfinite(V)
        u = U[mask]  # Δs
        v = V[mask]  # Δx
        speed = np.sqrt(u ** 2 + (x_scale * v) ** 2)
        # Note: quiver wants (dx, ds) = (Δx, Δs) because x is horizontal, s vertical.
        # _fixed_length_arrows(u, v) here takes (Δx, Δs) with spans (x_span, s_span).
        dx_hat, ds_hat = _fixed_length_arrows(
            v, u, s_span=x_span, x_span=s_span, length_frac=arrow_len
        )
        ax.quiver(
            X[mask],
            S[mask],
            dx_hat,
            ds_hat,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0045,
            headwidth=4.0,
            headlength=5.0,
            headaxislength=4.0,
            color=cmap(norm(speed)),
            alpha=0.95,
            zorder=3,
            pivot="tail",
        )
        ax.set_xlim(*x_lim)
        ax.set_ylim(*s_lim)
        ax.set_box_aspect(1)
        ax.set_xlabel(x_label)
        ax.set_title(name)
        ax.text(
            0.02,
            0.98,
            f"n_traj={n_seeds_by_panel[name]}\nmin_count={min_count}\n"
            f"{last_note}\n{smooth_note}\n"
            f"arrow bins: {n_x_bins}(x)×{n_s_bins}(s)\n"
            f"inset cells: {n_x_cells}(x)×{n_s_cells}(s)\n"
            f"insets: raw visit frac vs time (K={n_time_bins})\n"
            f"fixed arrow length",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="0.25",
            zorder=4,
        )

    axes[0].set_ylabel(s_label)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label(rf"scaled speed $\sqrt{{\Delta s^2 + ({x_scale:g}\Delta x)^2}}$")

    if title is None:
        title = (
            "Mean flow + per-cell visit time series\n"
            "arrows = direction of mean $(\\Delta s, \\Delta x)$ (fixed length); "
            "color = speed; "
            f"background = visit fraction vs time on {n_x_cells}×{n_s_cells} cells; "
            f"{last_note}; {smooth_note}"
        )
    fig.suptitle(title, fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _as_frame(df)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot mean-flow quiver + per-cell visit time-series mosaic "
            "from (t, x, s) CSV trajectories."
        )
    )
    p.add_argument(
        "--csv",
        action="append",
        required=True,
        help="CSV with columns t/time, x, s. Repeat for multiple trajectories "
        "(pooled as seeds of one panel). Use --panel to group into panels.",
    )
    p.add_argument(
        "--panel",
        action="append",
        default=None,
        help="Panel name for the corresponding --csv (same order). "
        "If omitted, all CSVs go into one panel.",
    )
    p.add_argument("--out", required=True, help="Output PNG path.")
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--smooth", type=int, default=40, dest="smooth_window")
    p.add_argument("--last-n", type=int, default=100_000, dest="last_n_steps")
    p.add_argument("--n-time-bins", type=int, default=5000)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--x-min", type=float, default=0.0)
    p.add_argument("--x-max", type=float, default=1.0)
    p.add_argument("--s-min", type=float, default=-0.5)
    p.add_argument("--s-max", type=float, default=9.5)
    p.add_argument("--x-scale", type=float, default=10.0)
    p.add_argument("--x-label", type=str, default="$x$")
    p.add_argument("--s-label", type=str, default="$s$")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--dpi", type=int, default=200)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    frames = [_load_csv(p) for p in args.csv]

    if args.panel is None:
        data: Any = frames
    else:
        if len(args.panel) != len(args.csv):
            raise SystemExit(
                f"--panel count ({len(args.panel)}) must match --csv count ({len(args.csv)})"
            )
        grouped: dict[str, list[pd.DataFrame]] = {}
        for name, fr in zip(args.panel, frames):
            grouped.setdefault(name, []).append(fr)
        data = grouped

    out = plot_mean_flow_visit_ts(
        data,
        args.out,
        n_bins=args.n_bins,
        smooth_window=args.smooth_window,
        last_n_steps=args.last_n_steps,
        n_time_bins=args.n_time_bins,
        min_count=args.min_count,
        x_lim=(args.x_min, args.x_max),
        s_lim=(args.s_min, args.s_max),
        x_scale=args.x_scale,
        x_label=args.x_label,
        s_label=args.s_label,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
