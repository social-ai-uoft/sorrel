"""
Generic rolling-window correlation between two aligned series (optionally smoothed).
Useful for patterns like ``system_analysis`` panel 1 (punishment vs social harm).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def rolling_window_correlation(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    window_size: int,
    step_size: int = 1,
    start_index: int = 0,
    smooth_x: int | None = None,
    smooth_y: int | None = None,
    min_periods: int | None = 1,
    method: Literal["pearson", "spearman"] = "pearson",
) -> dict[str, np.ndarray]:
    """
    Slide a window of length ``window_size`` along aligned ``x`` and ``y`` (after
    optional trim ``start_index:``, then optional rolling means), and compute
    correlation + two-sided p-value per window.

    ``window_center_index`` is in the **trimmed** series index (0 = first kept point).

    Parameters
    ----------
    x, y
        Same length 1d sequences (e.g. punishment and social harm ``Value`` columns).
    start_index
        Drop leading ``[0:start_index)`` from both series before smoothing/windows
        (same idea as ``social_harm[start_timepoint:]`` in ``system_analysis``).
    window_size
        Number of consecutive points in each window.
    step_size
        Advance the window start by this many points each iteration.
    smooth_x, smooth_y
        If set, apply ``rolling(..., min_periods=...)`` with that window before slicing.
    method
        ``pearson`` or ``spearman``.

    Returns
    -------
    dict with keys ``window_center_index``, ``correlation``, ``p_value`` (float arrays;
    entries are NaN where the window was skipped: flat series, too few finite points).
    """
    xa = np.asarray(pd.Series(x).astype(float).values, dtype=float)
    ya = np.asarray(pd.Series(y).astype(float).values, dtype=float)
    if xa.shape != ya.shape:
        raise ValueError("x and y must have the same length")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    xa = xa[start_index:]
    ya = ya[start_index:]
    n = xa.shape[0]
    mp = min_periods if min_periods is not None else 1

    if smooth_x is not None and smooth_x > 1:
        xa = pd.Series(xa).rolling(window=smooth_x, min_periods=mp).mean().values
    if smooth_y is not None and smooth_y > 1:
        ya = pd.Series(ya).rolling(window=smooth_y, min_periods=mp).mean().values

    corr_fn = pearsonr if method == "pearson" else spearmanr
    centers: list[int] = []
    corrs: list[float] = []
    pvals: list[float] = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        xw = xa[start:end]
        yw = ya[start:end]
        ok = np.isfinite(xw) & np.isfinite(yw)
        xw = xw[ok]
        yw = yw[ok]
        if xw.size < 2 or yw.size < 2:
            corrs.append(np.nan)
            pvals.append(np.nan)
            centers.append(start + window_size // 2)
            continue
        if np.std(xw) == 0 or np.std(yw) == 0:
            corrs.append(np.nan)
            pvals.append(np.nan)
            centers.append(start + window_size // 2)
            continue
        c, p = corr_fn(xw, yw)
        corrs.append(float(c))
        pvals.append(float(p))
        centers.append(start + window_size // 2)

    return {
        "window_center_index": np.asarray(centers, dtype=int),
        "correlation": np.asarray(corrs, dtype=float),
        "p_value": np.asarray(pvals, dtype=float),
    }


def plot_rolling_correlation(
    result: dict[str, np.ndarray],
    *,
    ax=None,
    title: str = "Rolling window correlation",
    xlabel: str = "Window center (index)",
    ylabel: str = "Correlation",
    alpha: float = 0.05,
    ylim: tuple[float, float] = (-1.0, 1.0),
    marker_size: float = 3.0,
):
    """
    Scatter correlation vs window center; red if p < alpha, blue otherwise.
    Returns ``(fig, ax)`` if ``ax`` is None, else ``(ax.figure, ax)``.
    """
    import matplotlib.pyplot as plt

    wc = result["window_center_index"]
    r = result["correlation"]
    p = result["p_value"]
    sig = p < alpha
    nonsig = ~sig & np.isfinite(p)
    sig = sig & np.isfinite(p)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(wc[nonsig], r[nonsig], "o", color="blue", markersize=marker_size, label=f"Nonsignificant (p ≥ {alpha})")
    ax.plot(wc[sig], r[sig], "o", color="red", markersize=marker_size, label=f"Significant (p < {alpha})")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ylim)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig, ax


def load_value_series(csv_path: str, max_rows: int | None = None) -> pd.Series:
    """Read ``Step`` / ``Value`` CSV and return ``Value`` indexed by step (optional cap)."""
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.iloc[:max_rows]
    s = df.set_index("Step")["Value"].astype(float)
    return s.sort_index()


__all__ = [
    "rolling_window_correlation",
    "plot_rolling_correlation",
    "load_value_series",
]
