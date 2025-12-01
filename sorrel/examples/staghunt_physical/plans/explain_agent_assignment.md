# How Agent Location Assignment Works (Current vs. Intended)

## Current Flow (BUGGY):

1. **Spawn Points Setup:**
   ```python
   spawn_points = sorted(...)  # [upper_spawn(row 6), lower_spawn(row 8)]
   desired_spawn = spawn_points[spawn_point_idx]  # Select upper or lower
   other_spawn = spawn_points[1 - spawn_point_idx]
   
   # Reorder so desired spawn is FIRST
   world.agent_spawn_points = [desired_spawn, other_spawn]
   ```

2. **Agent Override:**
   ```python
   override_agents([probe_agent.agent, dummy_agent])
   # agents[0] = probe_agent.agent (focal agent)
   # agents[1] = dummy_agent
   ```

3. **Environment Reset:**
   ```python
   reset() → calls _populate_from_ascii_map()
   ```

4. **Agent Placement (THE BUG):**
   ```python
   # Line 380 in env.py:
   chosen_positions = random.sample(world.agent_spawn_points, num_spawn_needed)
   for loc, agent in zip(chosen_positions, self.agents[:num_spawn_needed]):
       world.add(loc, agent)
   ```

## The Problem:
- `random.sample()` **randomly** selects spawn points
- Even though we put `desired_spawn` first in `world.agent_spawn_points`, 
  `random.sample()` ignores order!
- Result: **Focal agent may end up at wrong location**

## Intended Flow (FIXED):

1-3. Same as above

4. **Agent Placement (FIXED):**
   ```python
   # Deterministic placement by order
   chosen_positions = world.agent_spawn_points[:num_spawn_needed]
   for loc, agent in zip(chosen_positions, self.agents[:num_spawn_needed]):
       world.add(loc, agent)
   ```

## Correct Assignment Logic:

When `spawn_point_idx=0` (upper):
- `world.agent_spawn_points = [(6, 6, 1), (8, 6, 1)]`
- `agents = [probe_agent.agent, dummy_agent]`
- After fix: `agents[0]` → `(6, 6, 1)`, `agents[1]` → `(8, 6, 1)` ✓

When `spawn_point_idx=1` (lower):
- `world.agent_spawn_points = [(8, 6, 1), (6, 6, 1)]`
- `agents = [probe_agent.agent, dummy_agent]`
- After fix: `agents[0]` → `(8, 6, 1)`, `agents[1]` → `(6, 6, 1)` ✓

