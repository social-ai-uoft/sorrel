"""Utility functions for state-punishment analysis scripts.

Includes reactive-control analysis, trajectory loading, and phase portraits
extracted / extended from analysis_v2.ipynb.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
from scipy.stats import pearsonr, zscore

ArrayLike = Union[pd.Series, np.ndarray, Sequence[float]]


def _read_value_csv(path: str, name: str, xlim: Optional[int] = None) -> pd.Series:
    df = pd.read_csv(path)
    if xlim is not None:
        df = df.iloc[:xlim]
    if "Value" not in df.columns:
        raise ValueError(f"{path} does not contain a 'Value' column.")
    return df["Value"].rename(name)


def _safe_zscore(a):
    a = np.asarray(a, dtype=float)
    if np.any(~np.isfinite(a)) or np.nanstd(a) == 0:
        return None
    z = zscore(a)
    if np.any(~np.isfinite(z)):
        return None
    return z


def _rolling_corr(x, y, window_size, step_size) -> pd.DataFrame:
    rows = []
    n = min(len(x), len(y))

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        x_win = np.asarray(x[start:end], dtype=float)
        y_win = np.asarray(y[start:end], dtype=float)
        center = start + window_size // 2

        if (
            np.any(~np.isfinite(x_win))
            or np.any(~np.isfinite(y_win))
            or np.std(x_win) == 0
            or np.std(y_win) == 0
        ):
            coef, pval = np.nan, np.nan
        else:
            coef, pval = pearsonr(x_win, y_win)

        rows.append(
            {
                "window_center": center,
                "coef": coef,
                "pval": pval,
            }
        )

    return pd.DataFrame(rows)


def _rolling_regression(
    df,
    outcome,
    focal_predictor,
    controls=None,
    window_size=1000,
    step_size=100,
    standardize=True,
) -> pd.DataFrame:
    if controls is None:
        controls = []

    # avoid duplicate predictors
    predictors = []
    for p in [focal_predictor] + controls:
        if p not in predictors:
            predictors.append(p)

    needed_cols = [outcome] + predictors

    rows = []
    n = len(df)

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        win = df.iloc[start:end][needed_cols].dropna()
        center = start + window_size // 2

        if len(win) < max(30, len(predictors) + 5):
            rows.append(
                {
                    "window_center": center,
                    "coef": np.nan,
                    "pval": np.nan,
                    "r2": np.nan,
                    "n_obs": len(win),
                }
            )
            continue

        y = win[outcome].to_numpy(dtype=float)
        X_cols = []
        valid = True

        if standardize:
            y_z = _safe_zscore(y)
            if y_z is None:
                valid = False
            else:
                y = y_z

            for col in predictors:
                z = _safe_zscore(win[col].to_numpy(dtype=float))
                if z is None:
                    valid = False
                    break
                X_cols.append(z)
        else:
            if np.std(y) == 0 or np.any(~np.isfinite(y)):
                valid = False

            for col in predictors:
                arr = win[col].to_numpy(dtype=float)
                if np.std(arr) == 0 or np.any(~np.isfinite(arr)):
                    valid = False
                    break
                X_cols.append(arr)

        if not valid:
            rows.append(
                {
                    "window_center": center,
                    "coef": np.nan,
                    "pval": np.nan,
                    "r2": np.nan,
                    "n_obs": len(win),
                }
            )
            continue

        X = sm.add_constant(np.column_stack(X_cols))
        model = sm.OLS(y, X).fit()

        rows.append(
            {
                "window_center": center,
                "coef": model.params[1],  # focal predictor coefficient
                "pval": model.pvalues[1],  # focal predictor p-value
                "r2": model.rsquared,
                "n_obs": len(win),
            }
        )

    return pd.DataFrame(rows)


def _plot_rolling_stat(
    ax,
    result_df,
    title,
    ylabel="Coefficient",
    alpha=0.05,
    ylim=(-1, 1),
):
    df = result_df.copy()

    valid = np.isfinite(df["coef"]) & np.isfinite(df["pval"])
    sig = valid & (df["pval"] < alpha)
    nonsig = valid & (df["pval"] >= alpha)

    ax.scatter(
        df.loc[nonsig, "window_center"],
        df.loc[nonsig, "coef"],
        s=6,
        color="blue",
        label=f"p ≥ {alpha}",
    )
    ax.scatter(
        df.loc[sig, "window_center"],
        df.loc[sig, "coef"],
        s=8,
        color="red",
        label=f"p < {alpha}",
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Window center")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def system_analysis_reactive_control(
    f,
    include_controls=True,
    use_rate=False,
    smooth_social_harm_window=1,
    smooth_punishment_window=1,
    window_size=1000,
    step_size=100,
    lag=1,
    start_timepoint=0,
    xlim=None,
    alpha=0.05,
    standardize=True,
    make_plot=True,
    panel2_use_recent_avg_harm=False,
    panel2_recent_avg_window=100,
    save_plot_path: Optional[str] = None,
    show_plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reactive-control analysis.

    Variables:
        x_t = social harm
        s_t = punishment level

    Tests:

    1. corr(x_t, s_t)
       Expected: negative

    2. Harm level -> future punishment change:
       If panel2_use_recent_avg_harm=False:
           s[t+lag] - s[t] ~ x_t + s_t
       If panel2_use_recent_avg_harm=True:
           s[t+lag] - s[t] ~ recent_avg_x_t + s_t

       Expected coefficient: positive

    3. Punishment level -> future harm change:
       x[t+lag] - x[t] ~ s_t + x_t
       Expected coefficient on s_t: negative

    4. Punishment change -> future harm change:
       x[t+lag] - x[t] ~ s[t] - s[t-lag] + x_t + s_t
       Expected coefficient on past Δs: negative

    5. Harm change -> future punishment change:
       s[t+lag] - s[t] ~ x[t] - x[t-lag] + x_t + s_t
       Expected coefficient on past Δx: positive

    Notes:
        - Tests 4 and 5 control for BOTH current state values x_t and s_t.
        - Panel 2 can optionally use recent average harm instead of current harm.
        - If save_plot_path is set, the figure is written to disk (script-friendly).
    """

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    social_harm = _read_value_csv(
        os.path.join(f, "Mean_mean_social_harm_received_data.csv"),
        "social_harm",
        xlim=xlim,
    )

    resource_A = _read_value_csv(
        os.path.join(f, "Mean_mean_a_encounters_data.csv"),
        "resource_A",
        xlim=xlim,
    )
    resource_B = _read_value_csv(
        os.path.join(f, "Mean_mean_b_encounters_data.csv"),
        "resource_B",
        xlim=xlim,
    )
    resource_C = _read_value_csv(
        os.path.join(f, "Mean_mean_c_encounters_data.csv"),
        "resource_C",
        xlim=xlim,
    )
    resource_D = _read_value_csv(
        os.path.join(f, "Mean_mean_d_encounters_data.csv"),
        "resource_D",
        xlim=xlim,
    )
    resource_E = _read_value_csv(
        os.path.join(f, "Mean_mean_e_encounters_data.csv"),
        "resource_E",
        xlim=xlim,
    )

    punishment = _read_value_csv(
        os.path.join(f, "Global_average_punishment_level_data.csv"),
        "punishment",
        xlim=xlim,
    )

    resource_total = (
        resource_A + resource_B + resource_C + resource_D + resource_E
    ).rename("resource_total")

    data = pd.concat(
        [
            social_harm,
            punishment,
            resource_total,
            resource_A,
            resource_B,
            resource_C,
            resource_D,
            resource_E,
        ],
        axis=1,
    ).dropna()

    if use_rate:
        data["social_harm_raw"] = data["social_harm"]
        data["social_harm"] = data["social_harm"] / data["resource_total"].replace(
            0, np.nan
        )

    data = data.iloc[start_timepoint:].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Smooth base variables
    # ------------------------------------------------------------------
    data["x"] = (
        data["social_harm"]
        .rolling(
            window=smooth_social_harm_window,
            min_periods=smooth_social_harm_window,
        )
        .mean()
    )

    data["s"] = (
        data["punishment"]
        .rolling(
            window=smooth_punishment_window,
            min_periods=smooth_punishment_window,
        )
        .mean()
    )

    # ------------------------------------------------------------------
    # Panel 2 optional recent-average harm predictor
    # This is a level-like accumulated disorder variable.
    # ------------------------------------------------------------------
    data[f"recent_avg_x_T{panel2_recent_avg_window}"] = (
        data["x"]
        .rolling(
            window=panel2_recent_avg_window,
            min_periods=panel2_recent_avg_window,
        )
        .mean()
    )

    # ------------------------------------------------------------------
    # Future and past changes
    # ------------------------------------------------------------------
    data[f"future_dx_lag{lag}"] = data["x"].shift(-lag) - data["x"]
    data[f"future_ds_lag{lag}"] = data["s"].shift(-lag) - data["s"]

    data[f"past_dx_lag{lag}"] = data["x"] - data["x"].shift(lag)
    data[f"past_ds_lag{lag}"] = data["s"] - data["s"].shift(lag)

    results = []

    # ------------------------------------------------------------------
    # Test 1: level association
    # corr(x_t, s_t)
    # ------------------------------------------------------------------
    res_corr = _rolling_corr(
        x=data["x"].to_numpy(),
        y=data["s"].to_numpy(),
        window_size=window_size,
        step_size=step_size,
    )
    res_corr["test"] = "level_corr_x_s"
    res_corr["formula"] = "corr(x_t, s_t)"
    res_corr["expected_sign"] = "negative"
    results.append(res_corr)

    # ------------------------------------------------------------------
    # Test 2: harm level -> future punishment change
    # Either current harm x_t or recent average harm recent_avg_x
    # ------------------------------------------------------------------
    if panel2_use_recent_avg_harm:
        panel2_focal = f"recent_avg_x_T{panel2_recent_avg_window}"
        panel2_test_name = "recent_avg_x_to_future_ds"
        panel2_formula = (
            f"s[t+{lag}] - s[t] ~ mean(x over last {panel2_recent_avg_window})"
            + (" + s_t" if include_controls else "")
        )
        panel2_title = (
            f"Recent average harm → future punishment change\n"
            f"s[t+{lag}] - s[t] ~ mean(x[t-{panel2_recent_avg_window}+1:t])"
            + (" + s_t" if include_controls else "")
            + "\nExpected: positive"
        )
    else:
        panel2_focal = "x"
        panel2_test_name = "x_to_future_ds"
        panel2_formula = (
            f"s[t+{lag}] - s[t] ~ x_t" + (" + s_t" if include_controls else "")
        )
        panel2_title = (
            f"Harm level → future punishment change\n"
            f"s[t+{lag}] - s[t] ~ x_t"
            + (" + s_t" if include_controls else "")
            + "\nExpected: positive"
        )

    controls = ["s"] if include_controls else []
    res_panel2 = _rolling_regression(
        data,
        outcome=f"future_ds_lag{lag}",
        focal_predictor=panel2_focal,
        controls=controls,
        window_size=window_size,
        step_size=step_size,
        standardize=standardize,
    )
    res_panel2["test"] = panel2_test_name
    res_panel2["formula"] = panel2_formula
    res_panel2["expected_sign"] = "positive"
    results.append(res_panel2)

    # ------------------------------------------------------------------
    # Test 3: punishment level -> future harm change
    # x[t+lag] - x[t] ~ s_t + x_t
    # ------------------------------------------------------------------
    controls = ["x"] if include_controls else []
    res_s_to_future_dx = _rolling_regression(
        data,
        outcome=f"future_dx_lag{lag}",
        focal_predictor="s",
        controls=controls,
        window_size=window_size,
        step_size=step_size,
        standardize=standardize,
    )
    res_s_to_future_dx["test"] = "s_to_future_dx"
    res_s_to_future_dx["formula"] = (
        f"x[t+{lag}] - x[t] ~ s_t" + (" + x_t" if include_controls else "")
    )
    res_s_to_future_dx["expected_sign"] = "negative"
    results.append(res_s_to_future_dx)

    # ------------------------------------------------------------------
    # Test 4: past punishment change -> future harm change
    # x[t+lag] - x[t] ~ s[t] - s[t-lag] + x_t + s_t
    # ------------------------------------------------------------------
    controls = ["x", "s"] if include_controls else []
    res_past_ds_to_future_dx = _rolling_regression(
        data,
        outcome=f"future_dx_lag{lag}",
        focal_predictor=f"past_ds_lag{lag}",
        controls=controls,
        window_size=window_size,
        step_size=step_size,
        standardize=standardize,
    )
    res_past_ds_to_future_dx["test"] = "past_ds_to_future_dx"
    res_past_ds_to_future_dx["formula"] = (
        f"x[t+{lag}] - x[t] ~ s[t] - s[t-{lag}]"
        + (" + x_t + s_t" if include_controls else "")
    )
    res_past_ds_to_future_dx["expected_sign"] = "negative"
    results.append(res_past_ds_to_future_dx)

    # ------------------------------------------------------------------
    # Test 5: past harm change -> future punishment change
    # s[t+lag] - s[t] ~ x[t] - x[t-lag] + x_t + s_t
    # ------------------------------------------------------------------
    controls = ["x", "s"] if include_controls else []
    res_past_dx_to_future_ds = _rolling_regression(
        data,
        outcome=f"future_ds_lag{lag}",
        focal_predictor=f"past_dx_lag{lag}",
        controls=controls,
        window_size=window_size,
        step_size=step_size,
        standardize=standardize,
    )
    res_past_dx_to_future_ds["test"] = "past_dx_to_future_ds"
    res_past_dx_to_future_ds["formula"] = (
        f"s[t+{lag}] - s[t] ~ x[t] - x[t-{lag}]"
        + (" + x_t + s_t" if include_controls else "")
    )
    res_past_dx_to_future_ds["expected_sign"] = "positive"
    results.append(res_past_dx_to_future_ds)

    results_df = pd.concat(results, ignore_index=True)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if make_plot:
        fig, axes = plt.subplots(1, 5, figsize=(36, 6), sharey=True)

        plot_specs = [
            (
                "level_corr_x_s",
                "Level association\ncorr(x_t, s_t)\nExpected: negative",
                "Correlation",
            ),
            (
                panel2_test_name,
                panel2_title,
                "Beta",
            ),
            (
                "s_to_future_dx",
                f"Punishment level → future harm change\n"
                f"x[t+{lag}] - x[t] ~ s_t"
                + (" + x_t" if include_controls else "")
                + "\nExpected: negative",
                "Beta",
            ),
            (
                "past_ds_to_future_dx",
                f"Punishment change → future harm change\n"
                f"x[t+{lag}] - x[t] ~ s[t] - s[t-{lag}]"
                + (" + x_t + s_t" if include_controls else "")
                + "\nExpected: negative",
                "Beta",
            ),
            (
                "past_dx_to_future_ds",
                f"Harm change → future punishment change\n"
                f"s[t+{lag}] - s[t] ~ x[t] - x[t-{lag}]"
                + (" + x_t + s_t" if include_controls else "")
                + "\nExpected: positive",
                "Beta",
            ),
        ]

        for ax, (test_name, title, ylabel) in zip(axes, plot_specs):
            df_plot = results_df[results_df["test"] == test_name]
            _plot_rolling_stat(
                ax,
                df_plot,
                title=title,
                ylabel=ylabel,
                alpha=alpha,
                ylim=(-1, 1),
            )

        plt.tight_layout()
        if save_plot_path is not None:
            os.makedirs(os.path.dirname(save_plot_path) or ".", exist_ok=True)
            fig.savefig(save_plot_path, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return results_df, data


def annotate_run_metadata(results_df: pd.DataFrame, run_path: str) -> pd.DataFrame:
    """Attach CPC / model / study columns inferred from a run folder name."""
    out = results_df.copy()
    basename = os.path.basename(run_path.rstrip("/"))
    out["run_name"] = basename
    # Same convention as the notebook block: "'01_cpc' in f".
    out["use_cpc"] = "01_cpc" in run_path
    out["model_type"] = "ppo" if "ppo" in basename else "iqn"
    out["study_number"] = 1 if "study1" in basename else 2
    return out


def list_run_dirs(
    runs_dir: str,
    exclude_substrings: Optional[Sequence[str]] = None,
    skip_copy: bool = True,
) -> list[str]:
    """List run directories under ``runs_dir``, optionally filtering names."""
    exclude_substrings = list(exclude_substrings or [])
    names = sorted(os.listdir(runs_dir))
    paths = []
    for name in names:
        full = os.path.join(runs_dir, name)
        if not os.path.isdir(full):
            continue
        if skip_copy and "copy" in name.lower():
            continue
        if any(substr in name for substr in exclude_substrings):
            continue
        paths.append(full)
    return paths


def resolve_explicit_runs(
    run_specs: Sequence[str],
    analysis_root: Optional[str] = None,
    runs_dir: Optional[str] = None,
) -> list[str]:
    """Resolve an explicit list of run dirs / basenames to absolute paths.

    Each entry may be:
      - an absolute path to a run directory
      - a path relative to ``analysis_root``
      - a bare run folder name looked up under ``runs_dir`` (or
        ``analysis_root/runs_dir`` when ``runs_dir`` is relative)
    """
    root = Path(analysis_root).resolve() if analysis_root else Path.cwd().resolve()
    search_dir: Optional[Path] = None
    if runs_dir:
        search_dir = Path(runs_dir)
        if not search_dir.is_absolute():
            search_dir = (root / search_dir).resolve()

    resolved: list[str] = []
    missing: list[str] = []
    for spec in run_specs:
        spec = spec.strip()
        if not spec:
            continue
        cand = Path(spec)
        if cand.is_absolute() and cand.is_dir():
            resolved.append(str(cand.resolve()))
            continue
        rel = (root / cand).resolve()
        if rel.is_dir():
            resolved.append(str(rel))
            continue
        if search_dir is not None:
            under = (search_dir / cand.name).resolve()
            if under.is_dir():
                resolved.append(str(under))
                continue
        missing.append(spec)

    if missing:
        raise FileNotFoundError(
            "Could not resolve run path(s): " + ", ".join(missing)
        )
    return resolved


# ---------------------------------------------------------------------------
# Trajectory loading (punishment / harm / votes / resources)
# ---------------------------------------------------------------------------


def load_step_series(
    filepath: str,
    ref_steps,
    fill_value: float = 0.0,
) -> pd.Series:
    """Load a TensorBoard CSV aligned to ``ref_steps`` by the Step column.

    Missing steps (epochs where the metric was not logged, e.g. zero-count
    actions in older runs) are filled with ``fill_value`` instead of being
    interpolated, which prevents artificial inflation when summing across
    action types.
    """
    df = pd.read_csv(filepath)
    s = df.set_index("Step")["Value"]
    return s.reindex(ref_steps, fill_value=fill_value).reset_index(drop=True)


# NOTE (phase / mean-flow x-axis):
# Default behavioral observable x is resource-A *share*:
#   x = Mean_mean_a_encounters / sum_{r in a..e} Mean_mean_r_encounters
# not raw A encounters. Share is comparable across seeds/activity levels and
# matches the intended (s, preference) phase plane. Use x_mode="raw_a" only
# when you explicitly want absolute A extraction intensity.
RESOURCE_NAMES = ("a", "b", "c", "d", "e")
S_CSV_NAME = "Global_average_punishment_level_data.csv"
X_A_CSV_NAME = "Mean_mean_a_encounters_data.csv"


def load_s_x_series(
    run_dir: str,
    *,
    x_mode: str = "a_share",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load aligned (t, s, x) for phase / mean-flow stages.

    Parameters
    ----------
    x_mode
        ``"a_share"`` (default): x = A / total resource encounters.
        ``"raw_a"``: x = Mean_mean_a_encounters only.

    Returns
    -------
    t, s, x, x_label
    """
    s_path = os.path.join(run_dir, S_CSV_NAME)
    if not os.path.isfile(s_path):
        raise FileNotFoundError(f"Missing {S_CSV_NAME} in {run_dir}")
    ref_df = pd.read_csv(s_path)
    ref_steps = ref_df["Step"].values
    t = ref_steps.astype(float)
    s = ref_df["Value"].to_numpy(dtype=float)

    mode = (x_mode or "a_share").strip().lower()
    if mode in {"raw_a", "a", "resource_a", "absolute_a"}:
        a_path = os.path.join(run_dir, X_A_CSV_NAME)
        if not os.path.isfile(a_path):
            raise FileNotFoundError(f"Missing {X_A_CSV_NAME} in {run_dir}")
        x = load_step_series(a_path, ref_steps).to_numpy(dtype=float)
        return t, s, x, "raw Mean_mean_a_encounters"

    if mode not in {"a_share", "share", "fraction", "a_over_total"}:
        raise ValueError(
            f"Unknown x_mode={x_mode!r}; expected 'a_share' or 'raw_a'"
        )

    parts = []
    for name in RESOURCE_NAMES:
        path = os.path.join(run_dir, f"Mean_mean_{name}_encounters_data.csv")
        if not os.path.isfile(path):
            # Absent arms contribute 0 (e.g. k=2 runs without c/d/e logs).
            parts.append(np.zeros(len(ref_steps), dtype=float))
            continue
        parts.append(load_step_series(path, ref_steps).to_numpy(dtype=float))
    stack = np.vstack(parts)
    total = stack.sum(axis=0)
    a = stack[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(total > 0, a / total, np.nan)
    return t, s, x, "A / total resource encounters"


def load_run_trajectory(run_dir: str) -> Dict[str, pd.Series]:
    """Load core time series for one run folder, aligned to punishment steps."""
    ref_df = pd.read_csv(os.path.join(run_dir, "Global_average_punishment_level_data.csv"))
    ref_steps = ref_df["Step"].values
    state_punishment_level = ref_df["Value"].reset_index(drop=True)

    social_harm_received = load_step_series(
        os.path.join(run_dir, "Mean_mean_social_harm_received_data.csv"), ref_steps
    )
    reward_series = load_step_series(
        os.path.join(run_dir, "Mean_mean_individual_score_data.csv"), ref_steps
    )
    vote_increase = load_step_series(
        os.path.join(run_dir, "Global_vote_increase_data.csv"), ref_steps
    )
    vote_decrease = load_step_series(
        os.path.join(run_dir, "Global_vote_decrease_data.csv"), ref_steps
    )

    resource_names = list(RESOURCE_NAMES)
    resources = {
        name: load_step_series(
            os.path.join(run_dir, f"Mean_mean_{name}_encounters_data.csv"),
            ref_steps,
        )
        for name in resource_names
    }
    resource_total = sum(resources.values())
    # Same definition as phase/mean-flow default x (A share of total extraction).
    with np.errstate(invalid="ignore", divide="ignore"):
        resource_A_share = resources["a"] / resource_total.replace(0, np.nan)

    return {
        "state_punishment_level": state_punishment_level,
        "social_harm_received": social_harm_received,
        "reward_series": reward_series,
        "vote_increase": vote_increase,
        "vote_decrease": vote_decrease,
        "resource_A": resources["a"],
        "resource_B": resources["b"],
        "resource_C": resources["c"],
        "resource_D": resources["d"],
        "resource_E": resources["e"],
        "resource_total": resource_total,
        "resource_A_share": resource_A_share,
        "steps": pd.Series(ref_steps, name="Step"),
    }


def load_runs_trajectories(run_dirs: Sequence[str]) -> Dict[str, List[pd.Series]]:
    """Load trajectory metrics for many runs into parallel lists (notebook style)."""
    keys = [
        "state_punishment_level",
        "social_harm_received",
        "reward_series",
        "vote_increase",
        "vote_decrease",
        "resource_A",
        "resource_B",
        "resource_C",
        "resource_D",
        "resource_E",
        "resource_total",
        "resource_A_share",
        "steps",
    ]
    out: Dict[str, List[pd.Series]] = {k: [] for k in keys}
    for run_dir in run_dirs:
        traj = load_run_trajectory(run_dir)
        for k in keys:
            out[k].append(traj[k])
    return out


# ---------------------------------------------------------------------------
# Phase portraits (2D and 3D with time)
# ---------------------------------------------------------------------------


def _prepare_xy(
    x: ArrayLike,
    y: ArrayLike,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    time: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if start is None:
        start = 0
    if end is None:
        end = len(x)

    x = x[start:end:step]
    y = y[start:end:step]

    if time is None:
        # Absolute step/episode index after start/end/step slicing.
        t = np.arange(start, start + len(x) * step, step, dtype=float)[: len(x)]
    else:
        t = np.asarray(time, dtype=float)[start:end:step]
        if len(t) != len(x):
            raise ValueError("time must align with x/y after slicing.")

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    return x[mask], y[mask], t[mask]


def phase_portrait_xy(
    x: ArrayLike,
    y: ArrayLike,
    x_name: str = "s_t",
    y_name: str = "x_t",
    title: str = "Phase portrait",
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    arrows: bool = True,
    arrow_every: int = 50,
    color_by_time: bool = False,
    connect: bool = False,
    point_size: int = 10,
    line_alpha: float = 0.25,
    point_alpha: float = 0.5,
    arrow_scale: float = 1,
    save_path: Optional[str] = None,
    show: bool = True,
    ax=None,
):
    """2D phase portrait of (x, y), optionally colored / connected by time order."""
    x, y, t = _prepare_xy(x, y, start=start, end=end, step=step)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    if connect:
        ax.plot(x, y, linewidth=1, alpha=line_alpha)

    if color_by_time:
        sc = ax.scatter(x, y, c=t, s=point_size, alpha=point_alpha)
        fig.colorbar(sc, ax=ax, label="time / episode")
    else:
        ax.scatter(x, y, s=point_size, alpha=point_alpha)

    if arrows and len(x) > 1:
        idx = np.arange(0, len(x) - 1, arrow_every)
        dx = x[idx + 1] - x[idx]
        dy = y[idx + 1] - y[idx]
        ax.quiver(
            x[idx],
            y[idx],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=arrow_scale,
            width=0.006,
            alpha=0.8,
        )

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show and created_fig:
        plt.show()
    elif not show and created_fig:
        plt.close(fig)
    return fig, ax


def phase_portrait_xyt(
    x: ArrayLike,
    y: ArrayLike,
    time: Optional[ArrayLike] = None,
    x_name: str = "s_t",
    y_name: str = "x_t",
    t_name: str = "t (step / episode)",
    title: str = "3D phase portrait",
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    arrows: bool = False,
    arrow_every: int = 50,
    connect: bool = True,
    point_size: int = 8,
    line_alpha: float = 0.35,
    point_alpha: float = 0.65,
    elev: float = 22.0,
    azim: float = -60.0,
    save_path: Optional[str] = None,
    show: bool = True,
    ax=None,
):
    """3D phase portrait in (x, y, t) with time/episode as the third axis.

    When ``time`` is None, uses the sample index (after start/end/step slicing),
    which corresponds to episode/step order for these TensorBoard series.
    Direction arrows default to off (trajectory path + points only).
    """
    x, y, t = _prepare_xy(x, y, start=start, end=end, step=step, time=time)

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if connect and len(x) > 1:
        ax.plot(x, y, t, linewidth=1.0, alpha=line_alpha, color="gray")

    sc = ax.scatter(x, y, t, c=t, s=point_size, alpha=point_alpha, cmap="viridis")
    fig.colorbar(sc, ax=ax, shrink=0.7, label=t_name)

    if arrows and len(x) > 1:
        idx = np.arange(0, len(x) - 1, arrow_every)
        dx = x[idx + 1] - x[idx]
        dy = y[idx + 1] - y[idx]
        dt = t[idx + 1] - t[idx]
        ax.quiver(
            x[idx],
            y[idx],
            t[idx],
            dx,
            dy,
            dt,
            length=1.0,
            normalize=False,
            linewidth=0.6,
            arrow_length_ratio=0.15,
            alpha=0.7,
            color="black",
        )

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(t_name)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show and created_fig:
        plt.show()
    elif not show and created_fig:
        plt.close(fig)
    return fig, ax
