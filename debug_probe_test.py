"""Debug script to verify probe test spawn placement and orientation issues."""

import sys
sys.path.insert(0, '/Users/socialai3/Documents/GitHub/sorrel')

from sorrel.examples.staghunt_physical.probe_test import TestIntentionProbeTest
from sorrel.examples.staghunt_physical.agents_v2 import StagHuntAgent
from sorrel.examples.staghunt_physical.env import StagHuntEnv
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation import observation_spec
from sorrel.agents.action_spec import ActionSpec
import numpy as np

# Create a dummy agent for testing
entity_list = ["Empty", "Wall", "Spawn", "StagResource", "HareResource", 
               "StagHuntAgentNorth", "StagHuntAgentEast", "StagHuntAgentSouth", "StagHuntAgentWest",
               "Sand", "AttackBeam", "PunishBeam"]

observation_spec_obj = observation_spec.OneHotObservationSpec(
    entity_list, full_view=False, vision_radius=3
)
action_spec = ActionSpec(["FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT", "ATTACK"])

# Simple dummy model
class DummyModel:
    def get_action_with_qvalues(self, state):
        q_values = np.random.randn(len(action_spec.actions))
        action = np.argmax(q_values)
        return action, q_values
    def get_action(self, state):
        return 0
    def reset(self):
        pass
    @property
    def memory(self):
        return self

dummy_model = DummyModel()
agent = StagHuntAgent(
    observation_spec=observation_spec_obj,
    action_spec=action_spec,
    model=dummy_model,
    interaction_reward=1.0,
    max_health=5
)

# Create probe test instance
probe_test = TestIntentionProbeTest(output_dir="/tmp/debug_probe")

# Check spawn points
print("=" * 60)
print("DEBUGGING PROBE TEST SPAWN PLACEMENT")
print("=" * 60)

# Access the test environment
probe_test._setup_test_env()

print(f"\nSpawn points from world: {probe_test.probe_env.test_world.agent_spawn_points}")
print(f"Number of spawn points: {len(probe_test.probe_env.test_world.agent_spawn_points)}")

# Check spawn point sorting
spawn_points = sorted(
    probe_test.probe_env.test_world.agent_spawn_points,
    key=lambda pos: (pos[0], pos[1])
)
print(f"\nSorted spawn points (by row, then col):")
for i, sp in enumerate(spawn_points):
    print(f"  Index {i}: row={sp[0]}, col={sp[1]} ({'upper' if i == 0 else 'lower'})")

# Check agent initial orientation
print(f"\nAgent initial orientation: {agent.orientation} ({'NORTH' if agent.orientation == 0 else 'WEST' if agent.orientation == 3 else 'OTHER'})")
print(f"Orientation mapping: 0=North, 1=East, 2=South, 3=West")

# Check what STEP_LEFT/STEP_RIGHT do from initial orientation
orientation_names = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"}
init_orient = agent.orientation

# From the code: STEP_LEFT rotates counterclockwise, STEP_RIGHT clockwise
# When facing orientation O with vector (dy, dx):
# STEP_LEFT: (dx, -dy)
# STEP_RIGHT: (-dx, dy)

ORIENTATION_VECTORS = {
    0: (-1, 0),  # north (up)
    1: (0, 1),   # east (right)
    2: (1, 0),   # south (down)
    3: (0, -1),  # west (left)
}

dy, dx = ORIENTATION_VECTORS[init_orient]
step_left_vec = (dx, -dy)
step_right_vec = (-dx, dy)

# Map back to orientation
VECTOR_TO_ORIENTATION = {
    (-1, 0): 0,  # north
    (0, 1): 1,   # east
    (1, 0): 2,   # south
    (0, -1): 3,  # west
}

step_left_orient = VECTOR_TO_ORIENTATION[step_left_vec]
step_right_orient = VECTOR_TO_ORIENTATION[step_right_vec]

print(f"\nFrom initial orientation {orientation_names[init_orient]}:")
print(f"  STEP_LEFT moves to: {step_left_vec} → {orientation_names[step_left_orient]}")
print(f"  STEP_RIGHT moves to: {step_right_vec} → {orientation_names[step_right_orient]}")

# Verify map layout
print(f"\nMap layout (from test_intention.txt):")
print("  Row 5: hare (2)")
print("  Row 6: upper spawn (A)")
print("  Row 7: stag (1)")
print("  Row 8: lower spawn (A)")
print("  Row 9: hare (2)")

# Check if mapping makes sense
if spawn_points[0][0] == 6:  # Upper spawn at row 6
    print(f"\nUpper spawn at row {spawn_points[0][0]}:")
    print(f"  - Above (row {spawn_points[0][0]-1}): hare")
    print(f"  - Below (row {spawn_points[0][0]+1}): stag")
    print(f"  If agent faces {orientation_names[init_orient]}:")
    if init_orient == 0:  # NORTH
        print(f"    STEP_LEFT → {orientation_names[step_left_orient]} → faces WEST → stag/hare unclear")
        print(f"    STEP_RIGHT → {orientation_names[step_right_orient]} → faces EAST → stag/hare unclear")
    elif init_orient == 3:  # WEST
        print(f"    STEP_LEFT → {orientation_names[step_left_orient]} → faces SOUTH → stag (below)")
        print(f"    STEP_RIGHT → {orientation_names[step_right_orient]} → faces NORTH → hare (above)")

if len(spawn_points) > 1 and spawn_points[1][0] == 8:  # Lower spawn at row 8
    print(f"\nLower spawn at row {spawn_points[1][0]}:")
    print(f"  - Above (row {spawn_points[1][0]-1}): stag")
    print(f"  - Below (row {spawn_points[1][0]+1}): hare")
    print(f"  If agent faces {orientation_names[init_orient]}:")
    if init_orient == 0:  # NORTH
        print(f"    STEP_LEFT → {orientation_names[step_left_orient]} → faces WEST → stag/hare unclear")
        print(f"    STEP_RIGHT → {orientation_names[step_right_orient]} → faces EAST → stag/hare unclear")
    elif init_orient == 3:  # WEST
        print(f"    STEP_LEFT → {orientation_names[step_left_orient]} → faces SOUTH → hare (below)")
        print(f"    STEP_RIGHT → {orientation_names[step_right_orient]} → faces NORTH → stag (above)")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)
print("1. Fix random.sample() in env.py to use deterministic placement")
print("2. Verify/fix initial orientation (code vs comment mismatch)")
print("3. Add logging to verify actual agent positions after reset")
print("4. Re-verify mapping logic based on ACTUAL initial orientation")

