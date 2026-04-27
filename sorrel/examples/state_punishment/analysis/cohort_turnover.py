"""
Turnover-aligned trajectories for old survivors vs new agents.

**Import in notebooks** (avoids stale kernels stuck on an old
``turnover_cohort_vote_trajectories``):

    from cohort_turnover import plot_turnover_encounter_cohorts

Supports:
- Vote difference (↑−↓) from action-frequency CSVs.
- Resource encounter counts from ``Agent_{slot}_{resource}_encounters_data.csv``.

Relative time 0 is the replacement step. Windows use [turnover − X, turnover + Y]
(inclusive). Turnovers without a fully valid window inside [0, end_] are dropped.
Curves average within-turnover agent means, then mean across turnovers (equal weight
per turnover).
"""
from __future__ import annotations

import csv
import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def load_agent_vote(
    folder: str,
    agent_idx: int,
    action_idxs: Sequence[int],
    length: int,
) -> pd.Series:
    total = None
    for a in action_idxs:
        path = os.path.join(folder, f"Agent_{agent_idx}_action_freq_action_{a}_data.csv")
        df = pd.read_csv(path)
        vals = df.set_index("Step")["Value"]
        total = vals if total is None else total.add(vals, fill_value=0)
    return total.sort_index()


def load_agent_encounter(
    folder: str,
    slot: int,
    resource: str,
    length: int,
) -> pd.Series:
    """Load one resource-encounter series (``Agent_{slot}_{resource}_encounters_data.csv``)."""
    del length  # kept for API parity with notebook callers
    fpath = os.path.join(folder, f"Agent_{slot}_{resource}_encounters_data.csv")
    df = pd.read_csv(fpath)
    vals = df.set_index("Step")["Value"]
    return vals.sort_index()


def find_agent_names_csv(run_folder: str) -> str | None:
    base = os.path.basename(os.path.normpath(run_folder))
    rp = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.basename(rp) == "state_punishment":
            break
        rp = os.path.dirname(rp)
    anims_root = os.path.join(rp, "data", "anims")
    head = os.path.join(anims_root, base)
    for _ in range(5):
        f = os.path.join(head, "agent_generation_reference", "agent_names.csv")
        if os.path.isfile(f):
            return f
        parent = os.path.dirname(head)
        if os.path.normpath(parent) == os.path.normpath(anims_root):
            break
        head = parent
    return None


def parse_agent_names(csv_path: str) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    with open(csv_path, newline="") as fp:
        for row in csv.reader(fp):
            if len(row) < 3:
                continue
            try:
                agent, ancestor, step = int(row[0]), int(row[1]), int(row[2])
                rows.append((step, ancestor, agent))
            except ValueError:
                continue
    return sorted(rows)


def get_agent_statuses_v2(
    agent_names_csv: str,
) -> tuple[list[dict], dict[int, int]]:
    """
    Returns (events, name_to_slot): name_to_slot[agent_id] -> file slot (0–9).
    Replacements inherit the ancestor's slot.
    """
    parsed = parse_agent_names(agent_names_csv)
    if not parsed:
        return [], {}

    pop_by_step: dict[int, set[int]] = {}
    name_to_slot: dict[int, int] = {}
    for step, anc, agent in parsed:
        pop_by_step.setdefault(step, set()).add(agent)
        if agent not in name_to_slot:
            name_to_slot[agent] = name_to_slot.get(anc, anc)

    events: list[dict] = []
    prev_set: set[int] = set()
    for epoch in sorted(pop_by_step):
        arr_now = pop_by_step[epoch]
        new = arr_now - prev_set
        removed = prev_set - arr_now
        if new or removed or not prev_set:
            events.append(
                {
                    "step": epoch,
                    "present": frozenset(arr_now),
                    "removed": removed,
                    "new": new,
                    "old_survivors": arr_now & prev_set if prev_set else set(),
                }
            )
        prev_set = set(arr_now)
    return events, name_to_slot


def replacement_events_after_initial(event_list: list[dict]) -> list[dict]:
    """Steps with arrivals or departures, excluding the first snapshot (index 0)."""
    repl = [e for e in event_list if e["removed"] or e["new"]]
    return repl[1:] if len(repl) > 1 else []


def _first_removal_after(
    agent_id: int, turnover_step: int, event_list: list[dict]
) -> int | None:
    """Smallest event step ``> turnover_step`` where ``agent_id`` is removed, else None."""
    best: int | None = None
    for ev in event_list:
        if ev["step"] <= turnover_step:
            continue
        if agent_id not in ev["removed"]:
            continue
        st = ev["step"]
        if best is None or st < best:
            best = st
    return best


def _removal_after_map(
    agents: set[int], turnover_step: int, event_list: list[dict]
) -> dict[int, int | None]:
    return {a: _first_removal_after(a, turnover_step, event_list) for a in agents}


def _select_turnover_episodes(
    episodes: list[dict],
    *,
    max_turnover_events: int | None,
    turnover_slice: slice | None,
) -> list[dict]:
    """
    Subset replacement episodes in chronological order (after dropping the
    initial snapshot). If turnover_slice is set, it wins; else max_turnover_events
    keeps the first N; else all episodes are used.
    """
    if turnover_slice is not None:
        return episodes[turnover_slice]
    if max_turnover_events is not None:
        n = max(0, max_turnover_events)
        return episodes[:n]
    return episodes


def _smoothed_vote_diff(
    run_folder: str,
    slot: int,
    vote_increase_actions: Sequence[int],
    vote_decrease_actions: Sequence[int],
    length: int,
    end_: int,
    smooth_window: int,
) -> pd.Series | None:
    path = os.path.join(
        run_folder, f"Agent_{slot}_action_freq_action_1_data.csv"
    )
    if not os.path.isfile(path):
        return None
    try:
        inc = load_agent_vote(run_folder, slot, vote_increase_actions, length)
        dec = load_agent_vote(run_folder, slot, vote_decrease_actions, length)
        diff = (inc - dec).loc[:end_].fillna(0)
        return diff.rolling(window=smooth_window, min_periods=1).mean()
    except Exception:
        return None


def collect_agent_diffs(
    run_folder: str,
    agent_ids: Iterable[int],
    name_to_slot: dict[int, int],
    vote_increase_actions: Sequence[int],
    vote_decrease_actions: Sequence[int],
    length: int,
    end_: int,
    smooth_window: int,
) -> dict[int, pd.Series]:
    out: dict[int, pd.Series] = {}
    for aid in sorted(set(agent_ids)):
        slot = name_to_slot.get(aid, aid)
        s = _smoothed_vote_diff(
            run_folder,
            slot,
            vote_increase_actions,
            vote_decrease_actions,
            length,
            end_,
            smooth_window,
        )
        if s is not None:
            out[aid] = s
    return out


def _smoothed_encounter_series(
    run_folder: str,
    slot: int,
    resource: str,
    length: int,
    end_: int,
    smooth_window: int,
) -> pd.Series | None:
    path = os.path.join(
        run_folder, f"Agent_{slot}_{resource}_encounters_data.csv"
    )
    if not os.path.isfile(path):
        return None
    try:
        raw = load_agent_encounter(run_folder, slot, resource, length)
        s = raw.loc[:end_].fillna(0)
        return s.rolling(window=smooth_window, min_periods=1).mean()
    except Exception:
        return None


def collect_agent_encounters(
    run_folder: str,
    agent_ids: Iterable[int],
    name_to_slot: dict[int, int],
    resource: str,
    length: int,
    end_: int,
    smooth_window: int,
) -> dict[int, pd.Series]:
    out: dict[int, pd.Series] = {}
    for aid in sorted(set(agent_ids)):
        slot = name_to_slot.get(aid, aid)
        s = _smoothed_encounter_series(
            run_folder, slot, resource, length, end_, smooth_window
        )
        if s is not None:
            out[aid] = s
    return out


def _episode_cohort_mean(
    turnover_step: int,
    agents: set[int],
    agent_series: dict[int, pd.Series],
    window_before: int,
    window_after: int,
    end_: int,
    *,
    new_agent_cohort: bool = False,
    removal_after_step: dict[int, int | None] | None = None,
) -> pd.Series | None:
    """
    Mean per-agent series across agents, indexed by relative offset in
    [-window_before, window_after]. None if window not in [0, end_] or no agents.

    For ``new_agent_cohort=True``, absolute steps strictly before ``turnover_step``
    are set to NaN: new agents reuse the predecessor's file slot, so earlier rows
    in that CSV are not observations of the new identity.

    If ``removal_after_step`` maps an agent id to a step ``S``, absolute steps
    ``>= S`` are set to NaN: the slot CSV continues with a successor after that
    agent is removed, so long windows would otherwise mix identities (old and new
    cohorts).
    """
    lo = turnover_step - window_before
    hi = turnover_step + window_after
    if lo < 0 or hi > end_:
        return None

    segments: list[pd.Series] = []
    for agent_i in agents:
        series = agent_series.get(agent_i)
        if series is None:
            continue
        try:
            seg = series.loc[lo:hi]
        except Exception:
            continue
        if seg.shape[0] != window_before + window_after + 1:
            continue
        seg = seg.copy()
        if new_agent_cohort:
            mask_pre = seg.index < turnover_step
            if mask_pre.any():
                seg = seg.astype(float)
                seg.loc[mask_pre] = np.nan
        if removal_after_step:
            rs = removal_after_step.get(agent_i)
            if rs is not None:
                mask_post = seg.index >= rs
                if mask_post.any():
                    seg = seg.astype(float)
                    seg.loc[mask_post] = np.nan
        seg.index = seg.index.astype(int) - turnover_step
        segments.append(seg)

    if not segments:
        return None
    return pd.concat(segments, axis=1).mean(axis=1, skipna=True)


def aligned_turnover_cohort_means(
    run_folder: str,
    agent_names_csv: str,
    *,
    length: int,
    end_: int | None,
    vote_increase_actions: Sequence[int],
    vote_decrease_actions: Sequence[int],
    window_before: int,
    window_after: int,
    smooth_window: int = 1,
    max_turnover_events: int | None = None,
    turnover_slice: slice | None = None,
) -> dict:
    """
    Equal-weight mean across turnovers (each turnover: mean across agents in the
    cohort). New-agent cohorts mask absolute steps before each turnover; both
    cohorts mask absolute steps at/after each agent’s first removal after that
    turnover (slot file would otherwise show a successor in long windows).

    Turnover subset (chronological, after skipping the initial replacement row):
    - turnover_slice: e.g. slice(0, 8) for the first 8, slice(3, 6) for indices 3–5.
    - max_turnover_events: shorthand for slice(0, N); ignored if turnover_slice is set.
    """
    end = length if end_ is None else end_

    event_list, name_to_slot = get_agent_statuses_v2(agent_names_csv)
    episodes = replacement_events_after_initial(event_list)
    episodes = _select_turnover_episodes(
        episodes,
        max_turnover_events=max_turnover_events,
        turnover_slice=turnover_slice,
    )
    n_turnovers_after_selection = len(episodes)
    if not episodes:
        return {
            "offsets": np.arange(-window_before, window_after + 1, dtype=int),
            "old_survivors": None,
            "new_agents": None,
            "n_episodes_old": 0,
            "n_episodes_new": 0,
            "n_turnovers_old": 0,
            "n_turnovers_new": 0,
            "n_turnovers_after_selection": 0,
        }

    all_ids: set[int] = set()
    for ev in episodes:
        all_ids |= set(ev["old_survivors"])
        all_ids |= set(ev["new"])

    agent_diffs = collect_agent_diffs(
        run_folder,
        all_ids,
        name_to_slot,
        vote_increase_actions,
        vote_decrease_actions,
        length,
        end,
        smooth_window,
    )

    old_ep_means: list[pd.Series] = []
    new_ep_means: list[pd.Series] = []
    for ev in episodes:
        s = ev["step"]
        old_ag = set(ev["old_survivors"])
        new_ag = set(ev["new"])
        om = _episode_cohort_mean(
            s,
            old_ag,
            agent_diffs,
            window_before,
            window_after,
            end,
            removal_after_step=_removal_after_map(old_ag, s, event_list),
        )
        if om is not None:
            old_ep_means.append(om)
        nm = _episode_cohort_mean(
            s,
            new_ag,
            agent_diffs,
            window_before,
            window_after,
            end,
            new_agent_cohort=True,
            removal_after_step=_removal_after_map(new_ag, s, event_list),
        )
        if nm is not None:
            new_ep_means.append(nm)

    idx = np.arange(-window_before, window_after + 1, dtype=int)

    def combine(ep_means: list[pd.Series]) -> pd.Series | None:
        if not ep_means:
            return None
        mat = pd.concat(ep_means, axis=1)
        return mat.mean(axis=1, skipna=True).reindex(idx)

    n_old = len(old_ep_means)
    n_new = len(new_ep_means)
    return {
        "offsets": idx,
        "old_survivors": combine(old_ep_means),
        "new_agents": combine(new_ep_means),
        "n_episodes_old": n_old,
        "n_episodes_new": n_new,
        "n_turnovers_old": n_old,
        "n_turnovers_new": n_new,
        "n_turnovers_after_selection": n_turnovers_after_selection,
    }


def aligned_turnover_cohort_means_encounters(
    run_folder: str,
    agent_names_csv: str,
    *,
    resource: str,
    length: int,
    end_: int | None,
    window_before: int,
    window_after: int,
    smooth_window: int = 1,
    max_turnover_events: int | None = None,
    turnover_slice: slice | None = None,
) -> dict:
    """
    Same aggregation as ``aligned_turnover_cohort_means``, but per-agent series
    are smoothed resource encounter counts for ``resource`` (e.g. ``"a"``).
    """
    end = length if end_ is None else end_

    event_list, name_to_slot = get_agent_statuses_v2(agent_names_csv)
    episodes = replacement_events_after_initial(event_list)
    episodes = _select_turnover_episodes(
        episodes,
        max_turnover_events=max_turnover_events,
        turnover_slice=turnover_slice,
    )
    n_turnovers_after_selection = len(episodes)
    if not episodes:
        return {
            "offsets": np.arange(-window_before, window_after + 1, dtype=int),
            "old_survivors": None,
            "new_agents": None,
            "n_episodes_old": 0,
            "n_episodes_new": 0,
            "n_turnovers_old": 0,
            "n_turnovers_new": 0,
            "n_turnovers_after_selection": 0,
        }

    all_ids: set[int] = set()
    for ev in episodes:
        all_ids |= set(ev["old_survivors"])
        all_ids |= set(ev["new"])

    agent_enc = collect_agent_encounters(
        run_folder,
        all_ids,
        name_to_slot,
        resource,
        length,
        end,
        smooth_window,
    )

    old_ep_means: list[pd.Series] = []
    new_ep_means: list[pd.Series] = []
    for ev in episodes:
        s = ev["step"]
        old_ag = set(ev["old_survivors"])
        new_ag = set(ev["new"])
        om = _episode_cohort_mean(
            s,
            old_ag,
            agent_enc,
            window_before,
            window_after,
            end,
            removal_after_step=_removal_after_map(old_ag, s, event_list),
        )
        if om is not None:
            old_ep_means.append(om)
        nm = _episode_cohort_mean(
            s,
            new_ag,
            agent_enc,
            window_before,
            window_after,
            end,
            new_agent_cohort=True,
            removal_after_step=_removal_after_map(new_ag, s, event_list),
        )
        if nm is not None:
            new_ep_means.append(nm)

    idx = np.arange(-window_before, window_after + 1, dtype=int)

    def combine(ep_means: list[pd.Series]) -> pd.Series | None:
        if not ep_means:
            return None
        mat = pd.concat(ep_means, axis=1)
        return mat.mean(axis=1, skipna=True).reindex(idx)

    n_old = len(old_ep_means)
    n_new = len(new_ep_means)
    return {
        "offsets": idx,
        "old_survivors": combine(old_ep_means),
        "new_agents": combine(new_ep_means),
        "n_episodes_old": n_old,
        "n_episodes_new": n_new,
        "n_turnovers_old": n_old,
        "n_turnovers_new": n_new,
        "n_turnovers_after_selection": n_turnovers_after_selection,
    }


def plot_turnover_vote_cohorts(
    run_folder: str,
    *,
    length: int,
    end_: int | None,
    vote_increase_actions: Sequence[int],
    vote_decrease_actions: Sequence[int],
    window_before: int,
    window_after: int,
    smooth_window: int = 1,
    max_turnover_events: int | None = None,
    turnover_slice: slice | None = None,
    title_prefix: str = "",
    figsize: tuple[float, float] = (14, 12),
):
    """
    Two-panel plot: old survivors vs new agents, relative step on the x-axis.

    Subtitle ``(n=a/b turnovers with data)``: ``a`` = turnovers contributing to that
    panel’s curve; ``b`` = ``n_turnovers_after_selection`` from
    ``aligned_turnover_cohort_means`` (see ``max_turnover_events`` / ``turnover_slice``).
    """
    import matplotlib.pyplot as plt

    agent_csv = find_agent_names_csv(run_folder)
    if not agent_csv or not os.path.exists(agent_csv):
        raise FileNotFoundError(f"No agent_names.csv for run folder {run_folder!r}")

    res = aligned_turnover_cohort_means(
        run_folder,
        agent_csv,
        length=length,
        end_=end_,
        vote_increase_actions=vote_increase_actions,
        vote_decrease_actions=vote_decrease_actions,
        window_before=window_before,
        window_after=window_after,
        smooth_window=smooth_window,
        max_turnover_events=max_turnover_events,
        turnover_slice=turnover_slice,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    x = res["offsets"]

    n_sel = res.get("n_turnovers_after_selection", 0)

    def draw(
        ax,
        series: pd.Series | None,
        ylabel: str,
        panel_title: str,
        n_turnovers: int,
    ):
        if series is not None and series.notna().any():
            ax.plot(x, series.values, color="C0", lw=1.5, alpha=0.9)
        ax.axvline(0, color="red", linestyle=":", linewidth=2, alpha=0.7)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Relative step (0 = turnover)")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{title_prefix}{panel_title} (n={n_turnovers}/{n_sel} turnovers with data)".strip()
        )

    draw(
        ax1,
        res["old_survivors"],
        f"Mean vote ↑ − ↓ (smoothed), old survivors",
        "Old surviving agents",
        res["n_turnovers_old"],
    )
    draw(
        ax2,
        res["new_agents"],
        f"Mean vote ↑ − ↓ (smoothed), new agents",
        "New agents",
        res["n_turnovers_new"],
    )
    plt.tight_layout()
    return fig, (ax1, ax2), res


def plot_turnover_encounter_cohorts(
    run_folder: str,
    *,
    resource: str,
    length: int,
    end_: int | None,
    window_before: int,
    window_after: int,
    smooth_window: int = 1,
    max_turnover_events: int | None = None,
    turnover_slice: slice | None = None,
    title_prefix: str = "",
    figsize: tuple[float, float] = (14, 12),
):
    """
    Two-panel plot: old survivors vs new agents, encounter counts aligned on turnover.

    Subtitle ``(n=a/b turnovers with data)`` matches ``plot_turnover_vote_cohorts``.
    See ``aligned_turnover_cohort_means_encounters`` for selection / window args.
    """
    import matplotlib.pyplot as plt

    agent_csv = find_agent_names_csv(run_folder)
    if not agent_csv or not os.path.exists(agent_csv):
        raise FileNotFoundError(f"No agent_names.csv for run folder {run_folder!r}")

    res = aligned_turnover_cohort_means_encounters(
        run_folder,
        agent_csv,
        resource=resource,
        length=length,
        end_=end_,
        window_before=window_before,
        window_after=window_after,
        smooth_window=smooth_window,
        max_turnover_events=max_turnover_events,
        turnover_slice=turnover_slice,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    x = res["offsets"]
    r_up = resource.upper()
    n_sel = res.get("n_turnovers_after_selection", 0)

    def draw(
        ax,
        series: pd.Series | None,
        ylabel: str,
        panel_title: str,
        n_turnovers: int,
    ):
        if series is not None and series.notna().any():
            ax.plot(x, series.values, color="C0", lw=1.5, alpha=0.9)
        ax.axvline(0, color="red", linestyle=":", linewidth=2, alpha=0.7)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Relative step (0 = turnover)")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{title_prefix}{panel_title} (n={n_turnovers}/{n_sel} turnovers with data)".strip()
        )

    draw(
        ax1,
        res["old_survivors"],
        f"Mean resource {r_up} encounters (smoothed), old survivors",
        "Old surviving agents",
        res["n_turnovers_old"],
    )
    draw(
        ax2,
        res["new_agents"],
        f"Mean resource {r_up} encounters (smoothed), new agents",
        "New agents",
        res["n_turnovers_new"],
    )
    plt.tight_layout()
    return fig, (ax1, ax2), res


__all__ = [
    "load_agent_vote",
    "load_agent_encounter",
    "find_agent_names_csv",
    "parse_agent_names",
    "get_agent_statuses_v2",
    "replacement_events_after_initial",
    "collect_agent_diffs",
    "collect_agent_encounters",
    "aligned_turnover_cohort_means",
    "aligned_turnover_cohort_means_encounters",
    "plot_turnover_vote_cohorts",
    "plot_turnover_encounter_cohorts",
]
