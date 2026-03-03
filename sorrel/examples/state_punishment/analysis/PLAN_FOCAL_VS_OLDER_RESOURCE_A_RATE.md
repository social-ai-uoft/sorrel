# Plan: Focal Agent vs Older Agents — Resource A Rate in a Designated Window

## Goal

For every **focal agent** (each time a new agent enters the population), compare that agent’s resource A rate to the **average rate of agents older than it** over a **designated time range of X generations** after introduction. “X generations” means from the focal’s introduction epoch up to (but not including) the epoch of the X-th subsequent turnover — so the window **can extend beyond the next turnover** (e.g. X = 2 or 3 spans multiple turnover intervals). “Older” means agents whose entry epoch is strictly before the focal’s; we compare at each step within that window and aggregate **across all generations**.

---

## Relation to previous plan

- **Previous plan**: Focal vs **all other agents** at the same epoch, within each generation’s period (until next turnover).
- **This plan**: Focal vs **only agents older than the focal** (entry epoch &lt; focal’s entry epoch), in a **designated window of X generations** (epochs from introduction until the X-th subsequent turnover; can extend beyond the next turnover), aggregated across generations.

---

## Prerequisites: Data from Existing Cells

- **`agent_names_df`**: One row per `(epoch, agent_name)` with columns:
  - `epoch`, `agent_name`, `age`, `agent_id` (or `Ancestor_Name`)
  - `resource_A_consumption`, `resource_B_consumption`, …, `resource_E_consumption`
- **Agent entry epochs**: For each `agent_name`, the first epoch they appear (e.g. from rows where `age == 0`, or a precomputed `agent_entry_epochs` dict). Needed to define “older than focal.”
- **Resource A rate** (to be computed):  
  `resource_A_rate = resource_A_consumption / (resource_A + … + resource_E)`  
  per (epoch, agent), with division-by-zero guard.

---

## Step 1 — Compute resource A rate and entry epochs

- Add `resource_A_rate` to `agent_names_df` as in the previous plan (total consumption = sum A..E; rate = A / total; NaN if total is 0).
- Ensure **entry epoch per agent** is available: for each `agent_name`, entry_epoch = min(epoch) where that agent appears, or the epoch where `age == 0` for that agent. Store e.g. in a column `entry_epoch` (replicated per row) or in a dict `agent_entry_epochs`.

---

## Step 2 — Define focal agents and the designated time window (X generations)

- **Focal agents (generations)**: Same as before — each time a new agent enters (e.g. `age == 0`, `epoch > 0`, `epoch >= min_epoch`). Order by epoch then `agent_name` → list of (turnover_epoch, focal_agent_name). Each pair has a **generation_index** i (1, 2, …). We need the **ordered list of turnover epochs, one per generation**: the k-th element is the turnover epoch of the k-th focal (so T_i = turnover epoch of generation i). This list can have repeated epochs when multiple agents enter at the same epoch; T_{i+X} is then the turnover epoch of the (i+X)-th focal.
- **Designated window: X generations**. For focal with **generation_index i** (entered at **T_i**):
  - **Window** = all epochs from T_i up to but **not including** T_{i+X} (the epoch of the X-th turnover after this focal). So we include epochs e with **T_i ≤ e &lt; T_{i+X}**.
  - This can **extend beyond the next turnover** (e.g. X = 2 or 3 means we span 2 or 3 turnover intervals).
  - **Step** = epoch offset from introduction: step = e − T_i (step 0 = turnover epoch, step 1 = one epoch after, …). The number of steps in the window varies by focal (depends on how many epochs fall in [T_i, T_{i+X})).
  - If **i + X** exceeds the number of turnover epochs (e.g. focal is near the end of the run), define the window as [T_i, end of data] (use max_epoch + 1 or the last available epoch).
- So we do **not** use “only until next turnover”; we use **X generations** (up to X subsequent turnovers), which may span multiple turnover intervals.

---

## Step 3 — At each (generation, step): focal rate vs mean rate of “older” agents

At epoch **e = T_i + s** (turnover epoch of generation i plus step s):

1. **Focal agent’s rate**: `resource_A_rate` for (epoch = e, agent_name = focal_agent for generation i). If the focal agent is not present at e (e.g. replaced), skip this (generation_index, step) or treat as missing.
2. **“Older” agents at epoch e**: All agents present at epoch e whose **entry_epoch &lt; T_i** (they were in the population before the focal agent entered). Compute the **mean** of their `resource_A_rate` → `rate_older_mean`.
3. Record for (generation_index, step):
   - **diff** = focal rate − rate_older_mean  
   - **higher_than_older** = (focal rate &gt; rate_older_mean)

If there are **no** older agents at epoch e, `rate_older_mean` is undefined; skip that (generation_index, step) or store NaN and drop later.

---

## Step 4 — Build the raw results table

- **Raw result table**: One row per **(generation_index, epoch)** for every epoch **e** in the designated window [T_i, T_{i+X}) for that generation (and within the data). **Step** = e − T_i; the number of steps per generation varies (depends on the length of the X-generation interval).
- **Required columns**:
  1. **generation_index**
  2. **step** (0 = turnover epoch, 1 = one epoch after, …; max step varies by generation)
  3. **epoch**
  4. **diff** (focal rate − mean rate of older agents)
  5. **higher_than_older** (True/False)
- **Optional columns**: focal_agent_name, rate_focal, rate_older_mean, n_older (number of older agents at that epoch).
- Skip rows where there are no older agents at that epoch (or rate_older_mean is NaN).

---

## Step 5 — Summarize and visualize (across generations)

- **Summary statistics** (can be reported at step 0 only, or over the full window):
  - **At step 0 only**: Proportion of generations where focal’s A rate &gt; older mean; mean/median (and optionally std) of diff at introduction.
  - **Over the window**: Same metrics averaged over all (generation_index, step) rows in the table (optional).
- **Trajectory plot** (across generations):
  - **X-axis**: step (0, 1, 2, …). Different generations may have different max steps (X generations can span different numbers of epochs).
  - **Y-axis**: For each step s, **mean(diff)** over all generations that have a row at that step (listwise per step).
  - So each point = average (focal − older mean) at that step, across all focal agents that have data at that step; later steps may have fewer generations.
- **Optional**: Histogram of diff at step 0; proportion “focal higher” by step; number of generations contributing at each step.

---

## Step 6 — Implementation sketch

1. Reuse or recompute `resource_A_rate` and agent entry epochs from existing cells.
2. Build list of focal agents: (turnover_epoch, focal_agent_name), sorted by epoch then agent_name, with generation_index (1, 2, …). Build **ordered list of turnover epochs** (e.g. unique T_1 &lt; T_2 &lt; … or one per generation) so for each generation_index i we can get T_i and T_{i+X}. Set **num_generations** = X (e.g. 2 or 3).
3. For each (generation_index i, turnover_epoch T_i, focal_agent):
   - **Window end**: end_epoch = T_{i+X} if i+X exists in the turnover list, else max_epoch + 1 (or last epoch in data).
   - For each epoch e in [T_i, end_epoch) that exists in the data:
     - step = e − T_i.
     - Focal rate at (e, focal_agent). If missing, skip.
     - Older agents at e: rows with epoch == e and entry_epoch &lt; T_i. Mean(resource_A_rate) → rate_older_mean. If no older agents, skip.
     - Append row: generation_index, step, epoch=e, diff, higher_than_older, (optional: rate_focal, rate_older_mean, n_older).
4. Build DataFrame; optionally add focal_agent_name.
5. Summary: filter to step == 0 (or use full table); compute proportion higher, mean/median diff. Plot: groupby step → mean(diff), then plot step vs mean diff (listwise per step).

---

## Edge cases

- **No older agents at epoch e**: e.g. first epoch of the run, or all incumbents replaced. Skip that (generation_index, step) or store NaN and drop.
- **Focal agent not present at e**: If the focal was replaced before epoch e, either skip that step for this generation or treat as missing (listwise per step).
- **Multiple new agents at same epoch**: Each is a separate generation with the same turnover_epoch T. “Older” at e = T + s = agents with entry_epoch &lt; T (so neither of the two new agents counts as “older” for the other). Both focal agents get the same “older” set at each step. **Window**: If we use “one turnover epoch per generation”, the first of the two focals has window [T, T_{i+1}) = [T, T) = empty (because the next focal enters at the same T). Either document that the first focal gets no rows, or use a consistent window end for same-epoch focals (e.g. use the next *distinct* turnover epoch so all focals at T share the same window end and get a non-empty window).
- **Window beyond data**: If T_{i+X} or the end of the window is beyond max_epoch, cap the window at the data end; when averaging at each step, use only generations with data at that step (listwise per step).
- **Epoch 0**: Exclude from focal list (no “older” agents) or handle separately.

---

## Config parameters

- **min_epoch**: Only consider turnovers with epoch ≥ min_epoch (e.g. 10000).
- **num_generations** (X): Number of generations (turnover intervals) after the focal’s introduction to include. Window = [T_i, T_{i+X}). So X = 1 means “until the next turnover”; X = 2 or 3 extends beyond the next turnover. Same X for every focal.

---

## File and notebook placement

- Implement as new cell(s) in  
  `sorrel/examples/state_punishment/analysis/analysis.ipynb`,  
  after the cells that build `agent_names_df`, resource consumption, and (if used) agent entry epochs. Can follow the existing “New Agent vs Others” analysis cell.

---

## Review notes (consistency and edge cases)

- **Turnover list indexing**: Use one turnover epoch per focal (generation_index). So `turnover_epochs[k]` = epoch when the (k+1)-th focal entered; for focal i, T_i = turnover_epochs[i-1], T_{i+X} = turnover_epochs[i+X-1] when i+X ≤ number of focals. This keeps “X generations” meaning “X more focal agents”.
- **Entry epoch definition**: Unambiguous definition: for each `agent_name`, entry_epoch = min(epoch) over all rows where that agent appears (first appearance in the data). Use this for “older than focal” (entry_epoch &lt; T_i).
- **Same-epoch focals and empty window**: When two or more focals share the same T, the first focal’s window is [T_i, T_{i+1}) = [T, T) if the next focal also enters at T — so the first focal gets zero epochs. Decide explicitly: either (a) accept that and drop that focal from the results, or (b) define window end for same-epoch focals as the next *distinct* turnover epoch so all get a non-empty window.
- **Focal replaced during window**: Already handled: skip (generation_index, step) when the focal is not present at e; trajectory for that generation has fewer steps (listwise per step when plotting).
- **Older agents**: Only agents *present at epoch e* with entry_epoch &lt; T_i are counted; agents who left before e are not in the “older” set. So the comparison is “focal vs incumbents still present at e”.
