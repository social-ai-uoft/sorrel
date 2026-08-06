# Model audit

A read-through of everything in `sorrel/models/` — what's broken, what's wasteful, what
to clean up. This is a static review (June 2026), so "this crashes" means "this crashes
when you hit that path," not "we ran it." Anything marked *(check)* is inferred from the
code and worth confirming before acting on it.

## The big one: GPU is broken for both RL models

Neither IQN nor PPO actually runs on CUDA, and every example quietly passes
`device="cpu"`, so nobody's hit it.

- **IQN** moves the *networks* to the device but not the *batch* it samples in
  `train_step` — `states`, `actions`, etc. (`iqn.py:343-348`) stay on CPU, so the first
  forward pass on `cuda` is a device mismatch. The `return loss.detach().numpy()` at the
  end (`:412`) also dies on CUDA (needs `.cpu()`).
- **PPO** does the opposite: it moves the *inputs* to the device (`ppo.py:238-251`) but
  never moves the *policy network* (`self.policy = ActorCritic(...)`, `:184`).

One line each to fix, but worth a test that actually touches a GPU.

## IQN (`iqn.py`)

- `model_update_freq` does nothing. It's accepted, stored, and set to `4` by *every*
  example (treasurehunt, tag, threadsafe, cleanup config) — but never read. Training runs
  every step regardless. Honor it or delete it.
- Soft update *and* hard sync both run: Polyak `soft_update` every step (`:410`) plus a
  full `load_state_dict` every `sync_freq` epochs (`:435`). Pick one — the hard copy just
  wipes out the Polyak averaging.
- *(check)* The target discounts by `GAMMA ** n_step` (n_step=3) but the buffer returns a
  1-step reward/next-state, so the horizon doesn't match and the target is probably biased.
- NoisyNets *and* epsilon-greedy are both on, and the noise is off at action time anyway
  (`eval()` in `_q_values_from_policy`). Redundant exploration.
- `random.choices(np.arange(...), k=1)` for one action — `random.randrange(action_space)`.

## PPO (`ppo.py`)

- Runs in **float64** end to end (`ActorCritic.double()`, `:116`): double the memory, much
  slower, no benefit for RL. Should be float32.
- `end_epoch_action` crashes when an epoch had no terminal — `np.nonzero(dones)[0][0]`
  (`:211`) throws `IndexError` on an empty result.
- `train_step` trains on the *whole* pre-allocated buffer (zeros included) and trusts that
  `end_epoch_action` already truncated it. Wrong call order = training on zero rows. The
  truncation also keeps only the first episode and drops the rest.
- Advantage is just `returns - value` (REINFORCE baseline), not GAE. Works, but GAE is the
  usual PPO move.
- `Categorical` is built from a softmax head — `Categorical(logits=...)` is more stable.

## Threadsafe variants (`*_threadsafe.py`, `threadsafe_base_model.py`)

- The actor's `take_action` isn't locked and ignores the `PolicySnapshot` machinery that
  exists for exactly this, so an actor reading weights can race a learner mid-update. The
  buffer is threadsafe; the policy read isn't.
- The lock is built lazily on first use (`threadsafe_base_model.py:14`) — two threads
  hitting it first can race on creating the lock itself. Just make it in `__init__`.
- The snapshot deep-copies the whole module every time `_version` ticks, which is every
  train/epoch call — a full model copy per step.
- `ThreadsafeTransformerBuffer` only locks the base `sample`; the transformer actually
  calls `sample_transformer`, which isn't overridden or locked (`threadsafe/buffers.py:79`).
- PPO's threadsafe variant copy-pastes the truncation logic (`ppo_threadsafe.py:68`) —
  same `IndexError`, now in two places.

## Transformer (`transformer.py`)

This one handles devices correctly (everything `.to(device)` in `__init__`) — the RL
models could copy that. But:

- `TransformerBlock` defaults `attention_type="starformer"` and never passes it down
  (`:330`), so it always uses the regular attention. The starformer path is dead.
- `StarformerAttention` inherits a `MultiheadAttention` from its parent that it never uses
  — extra params that get trained and saved for nothing.
- `action_targets.squeeze()` (`:619`) eats a dimension when batch or time is 1.
- Lots of `torch.tensor(numpy_array)` in `get_batch` — copies and warns; `torch.from_numpy`.

## Interface (`base_model.py`, `pytorch_base.py`, `layers.py`)

- `@abstractmethod` is decorative — nothing is an `ABC`, so you can instantiate
  `BaseModel` and forget `take_action`. Make it `abc.ABC`.
- `take_action` is typed `-> int`, but PPO returns `(action, log_prob)`. That mismatch
  (the pyright warning) leaks everywhere — the buffer and the loop both special-case PPO.
  Worth a real action-return type before more models pile on.
- Small stuff: `seed == None` should be `is None` (`pytorch_base.py:42`); `NoisyLinear`
  crashes in `reset_parameters` if you pass `bias=False`.

## LLM (`llm.py`)

- `response.content[0].text` (`:98`) assumes a text block; anthropic can return other
  block types and this `AttributeError`s. (Same line as the pyright finding.)
- `take_action` does `action_list.index(output.lower())` — any stray word or punctuation
  from the model is a `ValueError`, and the OpenAI path can return `None` (so `.lower()`
  crashes). LLM output needs parsing, not a direct index lookup.
- `recall(method="frequency")` is a silent `pass`, `format_memories` prints, and the buffer
  shape is hardcoded to `[11, 11]`.

## HumanPlayer (`human_player.py`)

Fine as an interactive shim. Hardcoded `tile_size`/`num_channels` and tied to notebook
I/O, but it's not a learning model — low priority.

## Where to start

Cheap correctness wins first, roughly in order: the two device fixes, `model_update_freq`,
the transformer `attention_type`, and PPO's `end_epoch_action` guard. The interface work
(a real `ABC`, a proper `take_action` return type) is worth doing before new models get
built on top of it.
