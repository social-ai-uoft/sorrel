# Plan: New Agent vs Old Agents — Resource A Rate at Each Generation

## Goal

For every **generation** (each time a new agent enters the population), test whether the **new agent’s** rate of taking resource A is **higher than the average** rate of the **other (incumbent) agents** at that same epoch.

---

## Prerequisites: Data from Existing Cells

- **`agent_names_df`**: One row per `(epoch, agent_name)` with columns:
  - `epoch`, `agent_name`, `age`, `agent_id` (or `Ancestor_Name`)
  - `resource_A_consumption`, `resource_B_consumption`, …, `resource_E_consumption`
- **`agent_entry_epochs`**: Dict mapping each `agent_name` to the first epoch they appear (or infer from rows where `age == 0`).
- **Resource A rate** (to be computed):  
  `resource_A_rate = resource_A_consumption / (resource_A + resource_B + resource_C + resource_D + resource_E)`  
  for each agent at each epoch.

---

## Step 1 — Compute resource A rate per (epoch, agent)

- Add a column (e.g. `resource_A_rate`) to `agent_names_df`:
  - For each row, total consumption = sum of A, B, C, D, E.
  - `resource_A_rate = resource_A_consumption / total` (guard against division by zero if needed).
- Optionally restrict to `epoch > min_epoch` (e.g. 10000) and drop rows with missing consumption so rates are comparable.

---

## Step 2 — Define “generations” and periods between adjacent turnovers

- **Turnover events**: Epochs where a new agent enters (e.g. rows with `age == 0`, `epoch > 0`, `epoch >= min_epoch`). Order by epoch (and by `agent_name` if ties) → turnover epochs T_1 < T_2 < … < T_K.
- **Period i** (generation i): The interval from turnover i up to (but not including) the next turnover. So period 1 = epochs [T_1, T_2), period 2 = [T_2, T_3), …, period K = [T_K, end of data]. The **focal agent** for period i is the agent who entered at T_i.
- Optionally exclude `epoch == 0` so that “other agents” are clearly incumbents.

---

## Step 3 — For each period, at every epoch: focal agent’s rate vs others’ average

For each **period i** (focal agent = agent who entered at T_i) and for **every epoch e** in that period (T_i ≤ e < T_{i+1}, or T_i ≤ e ≤ max_epoch for the last period):

1. **Focal agent’s rate at e**: From `agent_names_df`, get `resource_A_rate` for (epoch = e, agent_name = focal).
2. **Other agents at epoch e**: All rows with `epoch == e` and `agent_name != focal`. Compute the **mean** of their `resource_A_rate` → `rate_others_mean`.
3. Record **both** for that (generation_index, epoch):
   - **Binary judgement**: `higher_than_others = (rate_focal > rate_others_mean)`
   - **Exact diff**: `diff = rate_focal - rate_others_mean`

---

## Step 4 — Build a results dataframe (raw table)

- **Raw result table**: One row per **(generation_index, epoch)** for every epoch in the period between two adjacent turnovers. So the table covers **all epochs** in each period, not only the turnover epoch.
- Required columns:
  1. **`generation_index`**: Period/turnover index (1, 2, 3, …). Period i = from turnover i until the next turnover.
  2. **`epoch`**: The epoch (every epoch within that period).
  3. **`diff`**: Exact difference (focal rate − others mean) at that epoch.
  4. **`higher_than_others`**: Binary judgement (True/False) at that epoch.
- Optional columns: `focal_agent_name`, `rate_new`, `rate_others_mean`.
- Skip rows where “others” is empty or rates are NaN.
- **Summary stats** (at turnover only): Filter the raw table to rows where `epoch == turnover_epoch` for that generation (i.e. step 0) to compute proportion higher, mean/median diff.
- **Trajectory plot**: For each generation_index, step s corresponds to epoch = T_i + s. From the raw table, take rows with that (generation_index, epoch); average `diff` across generations at each step (listwise per step).

---

## Step 5 — Summarize and visualize

- **Summary statistics** (use only rows where epoch = turnover epoch for that generation, i.e. step 0):
  - Proportion of generations where the new agent’s A rate is higher than the others’ average:  
    `higher_than_others.mean()` on the step-0 subset.
  - Mean / median of `diff` at turnover (and optionally std).
- **Plot — average trajectory of diff across generations**:
  - **Main plot**: Each point = **average diff at that step across generations**.
    - X-axis: **step** — step 0 = epoch when turnover happens (turnover epoch); step 1 = one epoch after turnover; step 2 = two epochs after; etc.
    - Y-axis: **mean(diff)** at that step, averaged over all generations that have data at that step. For each generation, the **focal agent** is the one who entered at that turnover; at each step s we compute diff = (focal agent’s resource_A_rate at turnover_epoch + s) − (mean resource_A_rate of *other* agents at that same epoch). Then average that diff across generations at each step.
  - At each step, average only over generations for which (turnover_epoch + step) is within the data (later generations may lack large steps; use listwise averaging per step).
  - Plot x = step, y = average diff at that step.
- **Optional extra plots**: Histogram of `diff` at step 0; proportion “new agent higher” by generation bins.

---

## Step 6 — Implementation sketch (single new cell)

1. Rely on `agent_names_df` and (if used) `agent_entry_epochs` from earlier cells.
2. Add `resource_A_rate` (and total consumption) to `agent_names_df` for the rows you need.
3. Build turnover epochs T_1, …, T_K (e.g. from rows with `age == 0`, `epoch > 0`, `epoch >= min_epoch`, sorted by epoch then agent_name). Define period i = [T_i, T_{i+1}) (or [T_K, max_epoch] for last).
4. For each period i and each epoch e in that period: focal = agent who entered at T_i; compute `rate_focal` and `rate_others_mean` at e; append row (generation_index=i, epoch=e, diff, higher_than_others). Build **raw results DataFrame** with one row per (generation_index, epoch).
5. **Summary stats**: Filter raw table to rows where epoch = that generation’s turnover epoch (step 0); compute proportion higher, mean/median diff. **Trajectory**: For each step s, from raw table take rows where epoch = T_i + s for each i; average `diff` across generations. **Plot**: x = step, y = average diff at that step.

---

## Edge cases

- **Epoch 0**: All agents are “new”; there are no “old” others. Either exclude from the “vs others” comparison or handle separately (e.g. label as “initial population”).
- **Zero total consumption** for an agent at an epoch: skip that row or set `resource_A_rate` to NaN and drop in the results.
- **Multiple new agents at the same epoch**: Treat each new agent as its own generation row; “others” for that row are all agents at that epoch except that one agent. For `generation_index`, use a tie-break (e.g. sort by epoch then by `agent_name`) so ordering is well-defined.
- **Trajectory steps near end of run**: For large step s, (turnover_epoch + s) may exceed the last epoch in the data. When averaging at each step, include only generations for which that epoch exists (listwise per step); the number of generations contributing to the mean may decrease for later steps.

---

## File and notebook placement

- Implement as one or more new cells in  
  `sorrel/examples/state_punishment/analysis/analysis.ipynb`,  
  after the cells that build `agent_names_df` and the resource consumption columns.

---

## Review notes (clarifications and edge cases)

- **Step 0 wording**: Step 0 is the epoch when turnover happens (turnover epoch); “first epoch after turnover” was clarified to avoid confusion with step 1.
- **Focal agent**: For the trajectory we follow **one agent per generation** — the one who entered at that turnover. At step 0 we compare that agent vs others at the turnover epoch. At step 1 we compare **that same agent** (now one epoch older) vs the other agents present at (turnover_epoch + 1). At step 2 we compare that same agent vs others at (turnover_epoch + 2), and so on. So we are not comparing a different “new” agent at each step; we track the agent who entered at turnover over time and at each step compute their diff vs the mean of everyone else at that epoch. “Others” = all agents at that epoch except this focal agent (the roster of “others” can change over steps as agents are replaced).
- **generation_index tie-break**: When multiple new agents share the same epoch, order by epoch then by a stable tie-break (e.g. `agent_name`) so `generation_index` is well-defined.
- **Trajectory averaging**: At each step, average diff only over generations that have data at (turnover_epoch + step); later steps may have fewer generations (listwise per step).
- **Summary stats**: All summary statistics use the results table, i.e. diff and higher_than_others at step 0 (turnover epoch).
