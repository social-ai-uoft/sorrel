# Continuity Mode Plan

## Goal

A new mode controlled by a single parameter. When **on**, three forms of continuity are preserved across epochs:

1. **Environment continuity** – The experiment/environment is not reset. No call to `self.reset()` after epoch 0, so all environment-level state (see below) continues across epoch boundaries.
2. **Spatial continuity** – World map, entity positions, agent positions, and resources stay as they are (this is the concrete consequence of not calling `reset()` and not repopulating the world).
3. **Temporal continuity** – Agent model memory (recurrent hidden state, replay buffers) is not cleared; `agent.model.start_epoch_action(epoch)` is not called at epoch start.

## What “environment continuity” preserves

When continuity mode is on and epoch > 0, we do **not** call `self.reset()`. So the following (which `reset()` would otherwise change) are preserved:

- **World:** `self.world.create_world()` and `self.populate_environment()` are not run, so the map, walls, spawn points, and current placement of entities (resources, agents) stay unchanged.
- **Environment state:** When starting the next segment we do **not** call `reset()`, but we **do** set `self.turn = 0` and `self.world.is_done = False` so the new epoch runs a segment of `current_epoch_max_turns` steps without the world being stuck in a "done" state. The world and agents are otherwise untouched.
- **Agents:** The loop `for agent in self.agents: agent.reset()` is not run, so agent state (inventory, orientation, cooldowns, health, removal/respawn state) and `model.reset()` are not called. So both env-level agent placement and per-agent state persist.

Thus “env continuity” means: the same environment instance keeps running with the same world and agent state; only the turn counter is reset for the next segment of steps.

## Parameter

- **Name:** `preserve_continuity`
- **Location:** `config["experiment"]["preserve_continuity"]`
- **Type:** `bool`, default `False`
- **Effect when True:**
  - **Environment continuity:** No `self.reset()` at the start of epochs after epoch 0. Environment and world state continue; only `self.turn` and `self.world.is_done` are set (turn = 0, is_done = False) for the next segment.
  - **Spatial continuity:** World map, agent positions, and resources are preserved (no `create_world()` / `populate_environment()` / agent placement reset).
  - **Temporal continuity:** No `agent.model.start_epoch_action(epoch=epoch)` at epoch start, so recurrent hidden state and replay buffers persist.

## Behaviour Summary

| Epoch   | Preserve continuity OFF (default)     | Preserve continuity ON                    |
|--------|---------------------------------------|-------------------------------------------|
| Start  | Always: full `reset()`, then `start_epoch_action()` | Epoch 0: full `reset()`. Epoch > 0: no reset; set `self.turn = 0` and `self.world.is_done = False`; no `start_epoch_action()` |
| Run    | Run until `turn >= current_epoch_max_turns` | Same (each epoch = another segment of `max_turns` steps in the same env/world) |
| End    | Train, log, metrics reset, etc.       | Same                                      |

## Implementation Points

1. **Config**
   - Add `preserve_continuity: false` under `experiment` in `main.py` (and any other config entry points).
   - Optional: CLI flag to override.

2. **Run loop (`env_with_probe_test.run_experiment`)**
   - At top of epoch loop: `preserve_continuity = self.config.experiment.get("preserve_continuity", False)`.
   - **Epoch start – “before reset” updates (dynamic_resource_density, appearance_switching):**
     - Run only when we are about to call `reset()`: `if not preserve_continuity or epoch == 0`.
   - **Epoch start – environment reset vs continuity:**
     - If `not preserve_continuity or epoch == 0`: call `self.reset()` (normal full-reset behaviour when continuity is OFF).
     - Else (continuity and epoch > 0): do **not** call `self.reset()`; set `self.turn = 0` and `self.world.is_done = False` so the next `while not self.turn >= self.current_epoch_max_turns` runs another segment of steps without touching the environment, world, or agents. (Setting `is_done = False` is required because the previous epoch set `world.is_done = True` at its end.)
   - **Epoch start – model (temporal continuity):**
     - If `not preserve_continuity`: call `agent.model.start_epoch_action(epoch=epoch)` for each spawned agent.
     - If `preserve_continuity`: skip this loop so model memory persists.
   - **Rest of loop:** unchanged (max_turns sampling, take_turn, epoch-end updates, train, log, probe test, save).

3. **Edge cases**
   - **Dynamic resource density:** When continuity is on and epoch > 0, we do not call `reset()` or `populate_environment()`, so `update_resource_density_at_epoch_start` has no effect until a future reset. Leaving the epoch-end density update as-is is fine (rates update for consistency; they only affect the next full reset).
   - **Appearance switching:** When continuity is on and epoch > 0, we skip the “before reset” block, so we do not call `switch_appearances()`. If appearance switches are needed in continuity mode, run `switch_appearances()` only when it does not assume a freshly populated world.
   - **Probe tests / model save:** No change; they operate on current agents and world.

4. **Base `environment.run_experiment`**
   - Not used by staghunt_physical’s main path (which uses `StagHuntEnvWithProbeTest`). If other callers use the base `run_experiment`, add the same parameter and logic there if they need continuity mode.

## Files to Touch

- `sorrel/examples/staghunt_physical/main.py`: add `preserve_continuity` to `config["experiment"]`.
- `sorrel/examples/staghunt_physical/env_with_probe_test.py`: implement the branching in `run_experiment()` as above.

## Testing

- Run with `preserve_continuity: false` (default): behaviour unchanged.
- Run with `preserve_continuity: true`: after epoch 0, confirm (1) environment continuity: no full reset, world and agents unchanged; (2) spatial: positions and resources persist; (3) temporal: model memory persists; (4) each epoch still runs `current_epoch_max_turns` steps and training/logging run every epoch.
