#!/usr/bin/env python3
"""
Standalone angular-consistency / net-rotation metrics for (s, x) phase trajectories.

s = Global_average_punishment_level
x = Mean_mean_a_encounters (or any resource series)

Primary API:
  phase_progression_metrics(s, x, ...) -> dict
  compute_metrics_table(records, ...) -> pandas.DataFrame
  plot_phase_diagnostics(t, s, x, out_path=None, ...)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ArrayLike = Union[Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------


def _as_1d(a: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return arr


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def _rolling_mean(y: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return y.copy()
    w = int(w)
    if w > len(y):
        return np.full_like(y, np.nan)
    kernel = np.ones(w, dtype=float) / w
    # same-length via edge padding
    pad = w // 2
    yp = np.pad(y, (pad, w - 1 - pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")


def _zscore(y: np.ndarray) -> np.ndarray:
    mu = np.nanmean(y)
    sd = np.nanstd(y)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(y)
    return (y - mu) / sd


def _detrend_1d(
    y: np.ndarray,
    *,
    mode: str = "none",
    rolling_w: Optional[int] = None,
) -> np.ndarray:
    """Remove slow drift from a 1-D series.

    Modes
    -----
    none : identity
    linear : subtract least-squares line (scipy.signal.detrend type)
    rolling : subtract centered rolling mean of width ``rolling_w``
    """
    y = np.asarray(y, dtype=float).copy()
    mode = (mode or "none").strip().lower()
    if mode in {"", "none", "off", "false", "0"}:
        return y
    if mode in {"linear", "lin", "l"}:
        n = len(y)
        if n < 2:
            return y
        tt = np.arange(n, dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return y
        coef = np.polyfit(tt[mask], y[mask], 1)
        return y - (coef[0] * tt + coef[1])
    if mode in {"rolling", "roll", "slow", "hp"}:
        w = int(rolling_w) if rolling_w is not None else max(3, len(y) // 10)
        w = max(3, w)
        trend = _rolling_mean(y, w)
        return y - trend
    raise ValueError(f"Unknown detrend mode={mode!r}; expected none|linear|rolling")


def phase_angle(s: ArrayLike, x: ArrayLike, *, standardize: bool = True) -> np.ndarray:
    """Return θ = atan2(x', s') in (-π, π], optionally after z-scoring each axis."""
    s = _as_1d(s, "s")
    x = _as_1d(x, "x")
    if len(s) != len(x):
        raise ValueError("s and x must have the same length")
    if standardize:
        s = _zscore(s)
        x = _zscore(x)
    return np.arctan2(x, s)


def unwrap_theta(theta: ArrayLike) -> np.ndarray:
    return np.unwrap(np.asarray(theta, dtype=float))


def angular_increments(theta_unwrapped: ArrayLike) -> np.ndarray:
    th = np.asarray(theta_unwrapped, dtype=float)
    if len(th) < 2:
        return np.array([], dtype=float)
    return np.diff(th)


def angular_consistency_D(dtheta: ArrayLike) -> float:
    """D = |sum e^{i Δθ}| / n  ∈ [0, 1]."""
    dth = np.asarray(dtheta, dtype=float)
    dth = dth[np.isfinite(dth)]
    n = len(dth)
    if n == 0:
        return float("nan")
    return float(np.abs(np.exp(1j * dth).sum()) / n)


def net_rotations_Nrot(dtheta: ArrayLike) -> float:
    """N_rot = sum(Δθ) / (2π)."""
    dth = np.asarray(dtheta, dtype=float)
    dth = dth[np.isfinite(dth)]
    if len(dth) == 0:
        return float("nan")
    return float(np.sum(dth) / (2.0 * np.pi))


def windowed_D(
    dtheta: ArrayLike, window: int, *, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window angular consistency.
    Returns (centers_index, D_values) where centers are indices into dtheta.
    """
    dth = np.asarray(dtheta, dtype=float)
    window = int(window)
    stride = max(1, int(stride))
    if window < 2 or len(dth) < window:
        return np.array([], dtype=int), np.array([], dtype=float)
    centers = []
    vals = []
    for start in range(0, len(dth) - window + 1, stride):
        chunk = dth[start : start + window]
        centers.append(start + window // 2)
        vals.append(angular_consistency_D(chunk))
    return np.asarray(centers, dtype=int), np.asarray(vals, dtype=float)


# ---------------------------------------------------------------------------
# Surrogates (optional)
# ---------------------------------------------------------------------------


def phase_randomized_surrogate(y: ArrayLike, rng: np.random.Generator) -> np.ndarray:
    """Fourier phase randomization preserving power spectrum (approx.)."""
    y = _as_1d(y, "y")
    n = len(y)
    y = y - np.nanmean(y)
    # Replace NaNs with 0 for FFT; rare after finite mask
    y = np.nan_to_num(y, nan=0.0)
    Y = np.fft.rfft(y)
    mag = np.abs(Y)
    # Keep DC and Nyquist phases fixed; randomize others
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=mag.shape))
    phases[0] = 1.0
    if n % 2 == 0:
        phases[-1] = 1.0
    return np.fft.irfft(mag * phases, n=n)


def surrogate_null_D(
    s: ArrayLike,
    x: ArrayLike,
    *,
    n_surr: int = 200,
    seed: int = 0,
    smooth_w: int = 1,
    standardize: bool = True,
) -> dict:
    """Null distribution of D under independent phase-randomized surrogates of s and x."""
    rng = np.random.default_rng(seed)
    s = _as_1d(s, "s")
    x = _as_1d(x, "x")
    Ds = []
    for _ in range(int(n_surr)):
        ss = phase_randomized_surrogate(s, rng)
        xx = phase_randomized_surrogate(x, rng)
        m = phase_progression_metrics(
            ss, xx, smooth_w=smooth_w, standardize=standardize, n_surr=0
        )
        Ds.append(m["D"])
    Ds = np.asarray(Ds, dtype=float)
    return {
        "D_surr": Ds,
        "D_surr_mean": float(np.nanmean(Ds)) if len(Ds) else float("nan"),
        "D_surr_p95": float(np.nanpercentile(Ds, 95)) if len(Ds) else float("nan"),
        "D_surr_p99": float(np.nanpercentile(Ds, 99)) if len(Ds) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main metric bundle
# ---------------------------------------------------------------------------


def phase_progression_metrics(
    s: ArrayLike,
    x: ArrayLike,
    *,
    t: Optional[ArrayLike] = None,
    last_n: Optional[int] = None,
    smooth_w: int = 1,
    standardize: bool = True,
    detrend: str = "linear",
    detrend_rolling_w: Optional[int] = None,
    windowed_w: Optional[int] = None,
    windowed_stride: int = 1,
    n_surr: int = 0,
    surr_seed: int = 0,
) -> dict:
    """
    Compute angular consistency / net rotation for one (s, x) trajectory.

    Parameters
    ----------
    s, x : series (same length)
    t : optional time/step axis
    last_n : if set, use only the last N samples after finite filtering
    smooth_w : rolling mean width applied to s and x before angle (1 = off)
    standardize : z-score s and x before atan2
    detrend : ``none`` | ``linear`` | ``rolling`` — applied after last_n,
        before smoothing. Default ``linear`` removes secular drift so
        D/N_rot reflect cycling more than slow walks.
    detrend_rolling_w : window for ``rolling`` detrend (default ~n/10)
    windowed_w : if set, also compute sliding-window D
    n_surr : if >0, also compute phase-randomized null for D
    """
    s = _as_1d(s, "s")
    x = _as_1d(x, "x")
    if len(s) != len(x):
        raise ValueError("s and x must have the same length")
    if t is None:
        t = np.arange(len(s), dtype=float)
    else:
        t = _as_1d(t, "t")
        if len(t) != len(s):
            raise ValueError("t must match s/x length")

    mask = _finite_mask(s, x, t)
    s, x, t = s[mask], x[mask], t[mask]

    if last_n is not None and last_n > 0 and len(s) > last_n:
        s, x, t = s[-last_n:], x[-last_n:], t[-last_n:]

    if len(s) < 3:
        return {
            "n": int(len(s)),
            "D": float("nan"),
            "N_rot": float("nan"),
            "D_smooth": float("nan"),
            "N_rot_smooth": float("nan"),
            "theta": np.array([], dtype=float),
            "theta_unwrapped": np.array([], dtype=float),
            "dtheta": np.array([], dtype=float),
            "t": t,
            "s_used": s,
            "x_used": x,
            "detrend": detrend,
        }

    # Order: window → detrend → smooth → (z-score inside phase_angle).
    s = _detrend_1d(s, mode=detrend, rolling_w=detrend_rolling_w)
    x = _detrend_1d(x, mode=detrend, rolling_w=detrend_rolling_w)

    s_s = _rolling_mean(s, smooth_w)
    x_s = _rolling_mean(x, smooth_w)

    theta = phase_angle(s, x, standardize=standardize)
    theta_u = unwrap_theta(theta)
    dth = angular_increments(theta_u)

    theta_s = phase_angle(s_s, x_s, standardize=standardize)
    theta_u_s = unwrap_theta(theta_s)
    dth_s = angular_increments(theta_u_s)

    out = {
        "n": int(len(s)),
        "D": angular_consistency_D(dth),
        "N_rot": net_rotations_Nrot(dth),
        "D_smooth": angular_consistency_D(dth_s),
        "N_rot_smooth": net_rotations_Nrot(dth_s),
        "theta": theta,
        "theta_unwrapped": theta_u,
        "dtheta": dth,
        "theta_smooth": theta_s,
        "theta_unwrapped_smooth": theta_u_s,
        "dtheta_smooth": dth_s,
        "t": t,
        "s_used": s,
        "x_used": x,
        "s_smooth": s_s,
        "x_smooth": x_s,
        "smooth_w": int(smooth_w),
        "standardize": bool(standardize),
        "detrend": (detrend or "linear"),
        "detrend_rolling_w": (
            None if detrend_rolling_w is None else int(detrend_rolling_w)
        ),
        "last_n": None if last_n is None else int(last_n),
    }

    if windowed_w is not None and windowed_w > 1:
        c_idx, d_win = windowed_D(dth_s, windowed_w, stride=windowed_stride)
        out["windowed_D_centers_idx"] = c_idx
        out["windowed_D"] = d_win
        # map centers to time (centers index into dtheta which aligns with t[1:])
        t_d = t[1:] if len(t) > 1 else t
        out["windowed_D_t"] = t_d[c_idx] if len(c_idx) else np.array([], dtype=float)

    if n_surr and n_surr > 0:
        null = surrogate_null_D(
            s,
            x,
            n_surr=n_surr,
            seed=surr_seed,
            smooth_w=smooth_w,
            standardize=standardize,
        )
        out.update(null)
        # one-sided p: fraction of surrogates with D >= observed
        obs = out["D_smooth"] if smooth_w and smooth_w > 1 else out["D"]
        out["D_surr_p"] = float(np.mean(null["D_surr"] >= obs))

    return out


def compute_metrics_table(
    records: Iterable[Mapping],
    *,
    s_key: str = "s",
    x_key: str = "x",
    t_key: str = "t",
    id_key: str = "run",
    group_key: Optional[str] = "group",
    last_n: Optional[int] = 20_000,
    smooth_w: int = 51,
    standardize: bool = True,
    detrend: str = "linear",
    detrend_rolling_w: Optional[int] = None,
    n_surr: int = 0,
    surr_seed: int = 0,
) -> pd.DataFrame:
    """
    records: iterable of dicts each containing at least s, x, and an id.
    Returns a tidy table of scalar metrics (one row per record).
    """
    rows = []
    for rec in records:
        s = rec[s_key]
        x = rec[x_key]
        t = rec.get(t_key)
        m = phase_progression_metrics(
            s,
            x,
            t=t,
            last_n=last_n,
            smooth_w=smooth_w,
            standardize=standardize,
            detrend=detrend,
            detrend_rolling_w=detrend_rolling_w,
            n_surr=n_surr,
            surr_seed=surr_seed,
        )
        row = {
            id_key: rec.get(id_key),
            "n": m["n"],
            "D": m["D"],
            "N_rot": m["N_rot"],
            "D_smooth": m["D_smooth"],
            "N_rot_smooth": m["N_rot_smooth"],
            "smooth_w": m["smooth_w"],
            "last_n": m["last_n"],
        }
        if group_key and group_key in rec:
            row[group_key] = rec[group_key]
        if n_surr and n_surr > 0:
            row["D_surr_mean"] = m.get("D_surr_mean")
            row["D_surr_p95"] = m.get("D_surr_p95")
            row["D_surr_p"] = m.get("D_surr_p")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_phase_diagnostics(
    t: ArrayLike,
    s: ArrayLike,
    x: ArrayLike,
    *,
    last_n: Optional[int] = 20_000,
    smooth_w: int = 51,
    standardize: bool = True,
    detrend: str = "linear",
    detrend_rolling_w: Optional[int] = None,
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> dict:
    """Three-panel diagnostic: phase plane, unwrapped θ/2π, Δθ histogram + metrics."""
    m = phase_progression_metrics(
        s,
        x,
        t=t,
        last_n=last_n,
        smooth_w=smooth_w,
        standardize=standardize,
        detrend=detrend,
        detrend_rolling_w=detrend_rolling_w,
    )
    tt = m["t"]
    ss = m["s_smooth"] if smooth_w and smooth_w > 1 else m["s_used"]
    xx = m["x_smooth"] if smooth_w and smooth_w > 1 else m["x_used"]
    th_u = (
        m["theta_unwrapped_smooth"]
        if smooth_w and smooth_w > 1
        else m["theta_unwrapped"]
    )
    dth = m["dtheta_smooth"] if smooth_w and smooth_w > 1 else m["dtheta"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # phase plane
    ax = axes[0]
    if standardize:
        ss_p, xx_p = _zscore(ss), _zscore(xx)
        ax.set_xlabel("z(s)")
        ax.set_ylabel("z(x)")
    else:
        ss_p, xx_p = ss, xx
        ax.set_xlabel("s")
        ax.set_ylabel("x")
    ax.plot(ss_p, xx_p, lw=0.8, alpha=0.85)
    ax.scatter([ss_p[0]], [xx_p[0]], c="green", s=28, zorder=3, label="start")
    ax.scatter([ss_p[-1]], [xx_p[-1]], c="red", s=28, zorder=3, label="end")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, loc="best")
    ax.set_title("phase plane")

    # unwrapped angle
    ax = axes[1]
    ax.plot(tt, th_u / (2 * np.pi), lw=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("θ / 2π (turns)")
    ax.set_title(
        f"unwrapped angle\nD={m['D_smooth']:.3f}  N_rot={m['N_rot_smooth']:.2f}"
    )

    # Δθ hist
    ax = axes[2]
    if len(dth):
        ax.hist(dth, bins=60, color="steelblue", alpha=0.85, density=True)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("Δθ (rad)")
    ax.set_ylabel("density")
    ax.set_title("angular increments")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return m


def plot_group_contrast(
    table: pd.DataFrame,
    *,
    group_col: str = "group",
    metric: str = "D_smooth",
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """Simple strip / box comparison of a metric across groups."""
    if group_col not in table.columns:
        raise KeyError(f"{group_col} not in table")
    groups = list(table[group_col].astype(str).unique())
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    data = [table.loc[table[group_col].astype(str) == g, metric].to_numpy() for g in groups]
    ax.boxplot(data, labels=groups, showfliers=False)
    for i, g in enumerate(groups):
        y = table.loc[table[group_col].astype(str) == g, metric].to_numpy()
        x = np.random.default_rng(0).normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, s=36, alpha=0.85, zorder=3)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by {group_col}")
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI (CSV convenience)
# ---------------------------------------------------------------------------


def _load_series_csv(path: Path, col: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a TensorBoard-style or plain CSV.
    Returns (t, y). If 'step'/'Wall time' present, use step as t.
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    t_col = None
    for cand in ("step", "steps", "t", "time", "wall time"):
        if cand in cols:
            t_col = cols[cand]
            break
    if col is not None:
        y_col = col
    else:
        # prefer Value, else last numeric column that isn't t
        if "value" in cols:
            y_col = cols["value"]
        else:
            num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if t_col in num:
                num.remove(t_col)
            if not num:
                raise ValueError(f"No numeric value column in {path}")
            y_col = num[-1]
    y = df[y_col].to_numpy(dtype=float)
    if t_col is not None:
        t = df[t_col].to_numpy(dtype=float)
    else:
        t = np.arange(len(y), dtype=float)
    return t, y


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Angular consistency D and net rotations N_rot for (s, x) trajectories."
    )
    p.add_argument("--s-csv", type=str, required=True, help="CSV for punishment / s series")
    p.add_argument("--x-csv", type=str, required=True, help="CSV for resource / x series")
    p.add_argument("--s-col", type=str, default=None)
    p.add_argument("--x-col", type=str, default=None)
    p.add_argument("--last-n", type=int, default=20_000)
    p.add_argument("--smooth-w", type=int, default=51)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--n-surr", type=int, default=0)
    p.add_argument("--out-metrics", type=str, default=None, help="Optional JSON/CSV path")
    p.add_argument("--out-plot", type=str, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    t_s, s = _load_series_csv(Path(args.s_csv), args.s_col)
    t_x, x = _load_series_csv(Path(args.x_csv), args.x_col)
    # align on shared length / overlapping steps if both have step
    n = min(len(s), len(x))
    s, x = s[:n], x[:n]
    t = t_s[:n] if len(t_s) >= n else np.arange(n, dtype=float)

    m = plot_phase_diagnostics(
        t,
        s,
        x,
        last_n=args.last_n,
        smooth_w=args.smooth_w,
        standardize=not args.no_standardize,
        title=args.title,
        out_path=args.out_plot,
        show=args.show,
    )
    if args.n_surr > 0:
        null = surrogate_null_D(
            m["s_used"],
            m["x_used"],
            n_surr=args.n_surr,
            smooth_w=args.smooth_w,
            standardize=not args.no_standardize,
        )
        m.update(null)
        obs = m["D_smooth"]
        m["D_surr_p"] = float(np.mean(null["D_surr"] >= obs))

    summary = {
        "n": m["n"],
        "D": m["D"],
        "N_rot": m["N_rot"],
        "D_smooth": m["D_smooth"],
        "N_rot_smooth": m["N_rot_smooth"],
    }
    if "D_surr_p" in m:
        summary["D_surr_mean"] = m["D_surr_mean"]
        summary["D_surr_p95"] = m["D_surr_p95"]
        summary["D_surr_p"] = m["D_surr_p"]

    print("Phase progression metrics")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.out_metrics:
        out = Path(args.out_metrics)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".json":
            import json

            with open(out, "w") as f:
                json.dump(summary, f, indent=2)
        else:
            pd.DataFrame([summary]).to_csv(out, index=False)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
