# Factored actions in bandit mode — plan, review, implementation

## 1. Goal

Allow **`iqn_use_factored_actions` / `ppo_use_factored_actions`** with **`iqn_action_dims` / `ppo_action_dims`** in **bandit** runs, using the same PyTorch IQN / PPO-LSTM machinery as the grid experiment, without silent shape or indexing bugs.

## 2. Plan

1. **Remove** the blanket `ValueError` in `validation.py` that rejected all bandit + factored configs.
2. **Validate** that the factored product matches the **bandit flat action count** implied by the experiment flags.
3. **Leave** `StatePunishmentBanditEnv._build_agent` as-is: it already wires `use_factored_actions`, `action_dims`, and `action_spec.n_actions`, and already checks `prod(action_dims) == action_spec.n_actions`.
4. **Leave** `BanditStatePunishmentAgent` decode unchanged: models’ `take_action` already returns a **single flat index**; factoring is internal to the network.

## 3. Review (constraints discovered)

### 3.1 Flat index is already unified

`PyTorchIQN.take_action` / `RecurrentPPOLSTM.take_action` combine factor heads into one `int` using the same mixed-radix style as training targets. `BanditStatePunishmentAgent._execute_bandit_core` only sees that **flat** `action`, same as unfactored runs.

### 3.2 Simple vs composite action counts

- **Simple** bandit: `n_actions = K + 3` (`K` picks, vote increase, vote decrease, noop), with `K = bandit_arms_per_trial`.
- **Composite** bandit: `n_actions = K*3 + 1` (pairs `move*3+vote` plus one noop index `K*3`).

### 3.3 Why composite + standard 2-head factored is not enabled

IQN’s two-head flatten uses `flat = move * action_dims[1] + vote` when `len(action_dims) == 2`. That matches composite decode **only if** the second dimension is **3** (vote factor) and the flat space is exactly **`move_dim * 3`**. Composite bandit also needs a **dedicated noop** at index **`K*3`**, so the flat size is **`K*3 + 1`**, not a full **`(K+1) * 3`** rectangle. There is no pair of positive integers `(d0, d3)` with `d1 = 3` whose product is `K*3+1`, because `K*3+1 ≡ 1 (mod 3)` while `d0*3 ≡ 0 (mod 3)`.

So **v1 policy**: allow factored only when **`use_composite_actions` is False** (simple bandit), and require **`prod(action_dims) == K + 3`**.

Users who need composite + structured heads should use **non-factored** flat softmax over `K*3+1` actions, or **dual-head PPO** on the grid path pattern (not built in `StatePunishmentBanditEnv` for bare `ppo`).

### 3.4 Responsibility for semantic layout

Even when `prod(dims) == K+3`, different `dims` tuples imply **different** mixed-radix orderings of the same flat indices. The agent’s **simple** decode is fixed (indices `0..K-1` arms, `K` / `K+1` votes, `K+2` noop). Users must choose **`iqn_action_dims` / `ppo_action_dims`** so that the model’s flattening matches the **intended** ordering of those slots (same issue as on the grid when using factored simple actions).

## 4. Implementation (done)

- **`validation.py`**: Removed unconditional factored bans; added checks:
  - composite + factored → clear `ValueError`;
  - factored → required dims string for the active model type;
  - `prod(dims) == bandit_arms_per_trial + 3` for simple bandit.
- **`BANDIT_MODE_PLAN.md`**: Updated limitations / factored subsection to match this behavior.

No changes were required in **`agents.py`** or **`env.py`** model construction for the supported case: existing code paths already match the grid factored wiring.
