
import unittest
import numpy as np
from sorrel.agents import MovingAgent
from sorrel.action.action_spec import ActionSpec
from sorrel.observation.observation_spec import ObservationSpec
from sorrel.worlds import Gridworld
from sorrel.entities import Entity
from sorrel.environment import Environment
from sorrel.models import BaseModel

# Mock classes for testing
class MockModel(BaseModel):
    def __init__(self, fixed_action=0):
        super().__init__(input_size=1, action_space=4, memory_size=10)
        self.fixed_action = fixed_action
        
    def take_action(self, state):
        return self.fixed_action
        
    def reset(self): pass
    def save(self, path): pass
    def load(self, path): pass
    def train_step(self): return 0

class MockAgent(MovingAgent):
    def __init__(self, action_spec, model, location=None):
        # Mock observation spec
        obs_spec = ObservationSpec(entity_list=["EmptyEntity"], full_view=True, env_dims=(5,5,1))
        super().__init__(obs_spec, action_spec, model, location)
        
    def pov(self, world):
        return np.array([0])
        
    def get_action(self, state):
        return self.model.take_action(state)
        
    def is_done(self, world):
        return False

class PassableEntity(Entity):
    def __init__(self):
        super().__init__()
        self.passable = True

class MockWorld(Gridworld):
    def __init__(self, height=5, width=5):
        super().__init__(height, width, 1, PassableEntity())

class MockEnv(Environment):
    def __init__(self, world, agents, simultaneous_moves=False):
        self.agents = agents
        # Minimal config
        config = {"experiment": {"epochs": 1, "max_turns": 1, "record_period": 1}}
        super().__init__(world, config, simultaneous_moves=simultaneous_moves)
        
    def setup_agents(self):
        # Already passed in init
        pass
        
    def populate_environment(self):
        for agent in self.agents:
            if agent._location:
                self.world.add(agent._location, agent)

class TestSimultaneousLogic(unittest.TestCase):
    def setUp(self):
        self.action_spec = ActionSpec(["up", "down", "left", "right"])
        
    def test_sequential_collision(self):
        """
        Test 1: Sequential mode.
        Agent 1 at (0,0) moves Right -> (0,1)
        Agent 2 at (0,2) moves Left -> (0,1)
        
        In sequential:
        - Agent 1 moves to (0,1).
        - Agent 2 tries to move to (0,1), finds it occupied by Agent 1, and fails (stays at 0,2).
        """
        world = MockWorld()
        
        # Agent 1: Right (index 3)
        a1 = MockAgent(self.action_spec, MockModel(3), location=(0,0,0))
        # Agent 2: Left (index 2)
        a2 = MockAgent(self.action_spec, MockModel(2), location=(0,2,0))
        
        env = MockEnv(world, [a1, a2], simultaneous_moves=False)
        
        # Before turn
        self.assertEqual(a1.location, (0,0,0))
        self.assertEqual(a2.location, (0,2,0))
        
        env.take_turn()
        
        # After turn
        # A1 should have moved to (0,1,0)
        self.assertEqual(a1.location, (0,1,0))
        # A2 tried to move to (0,1,0) but A1 was there, so blocked.
        self.assertEqual(a2.location, (0,2,0))

    def test_simultaneous_collision(self):
        """
        Test 2: Simultaneous mode.
        Agent 1 at (0,0) moves Right -> (0,1)
        Agent 2 at (0,2) moves Left -> (0,1)
        
        In simultaneous:
        - Both propose (0,1).
        - Conflict detected.
        - Neither moves.
        """
        world = MockWorld()
        
        # Agent 1: Right (index 3)
        a1 = MockAgent(self.action_spec, MockModel(3), location=(0,0,0))
        # Agent 2: Left (index 2)
        a2 = MockAgent(self.action_spec, MockModel(2), location=(0,2,0))
        
        env = MockEnv(world, [a1, a2], simultaneous_moves=True)
        
        env.take_turn()
        
        # Both should stay put
        self.assertEqual(a1.location, (0,0,0))
        self.assertEqual(a2.location, (0,2,0))

    def test_simultaneous_no_collision(self):
        """
        Test 3: Simultaneous mode, no collision.
        Agent 1 at (0,0) moves Down -> (1,0)
        Agent 2 at (0,2) moves Down -> (1,2)
        
        Both should move.
        """
        world = MockWorld()
        
        # Agent 1: Down (index 1)
        a1 = MockAgent(self.action_spec, MockModel(1), location=(0,0,0))
        # Agent 2: Down (index 1)
        a2 = MockAgent(self.action_spec, MockModel(1), location=(0,2,0))
        
        env = MockEnv(world, [a1, a2], simultaneous_moves=True)
        
        env.take_turn()
        
        self.assertEqual(a1.location, (1,0,0))
        self.assertEqual(a2.location, (1,2,0))

    def test_simultaneous_swap_attempt(self):
        """
        Test 4: Simultaneous swap (Head-on collision?)
        Agent 1 at (0,0) moves Right -> (0,1)
        Agent 2 at (0,1) moves Left -> (0,0)
        
        Destinations are distinct:
        A1 -> (0,1) (currently A2)
        A2 -> (0,0) (currently A1)
        
        In our logic:
        destinations[0,1] = [A1]
        destinations[0,0] = [A2]
        
        No destination conflict.
        So both are allowed.
        
        However, does world.move allow moving into an occupied square?
        world.move checks `self.map[new_location].passable`.
        If A2 is at (0,1) and is not passable (Agents usually aren't?), then A1's move will fail in `finalize_turn` when calling `world.move`.
        Same for A2 moving to (0,0).
        
        So even if simultaneous logic allows it, the physics engine (world.move) might block it if the space isn't cleared yet.
        Since we update sequentially in finalize_turn loop:
        - A1 tries to move to (0,1). A2 is there. Blocked?
        - A2 tries to move to (0,0). A1 is there (or moved?).
        
        This depends on if we remove agents before moving them?
        No, `finalize_turn` calls `world.move` one by one.
        So they will block each other unless we clear the board first.
        
        The requirement was: "if two agents simulatanously try to move into the same square, neither move into it."
        It didn't explicitly say "swapping is allowed".
        Given the current implementation, swapping will likely fail due to physical obstruction, which is fine/safe.
        Let's verify that they don't disappear or do something weird.
        """
        world = MockWorld()
        
        # Agent 1: Right (index 3)
        a1 = MockAgent(self.action_spec, MockModel(3), location=(0,0,0))
        # Agent 2: Left (index 2)
        a2 = MockAgent(self.action_spec, MockModel(2), location=(0,1,0))
        
        # Ensure agents are impassable (default for Entity is passable=False? No, Entity() is passable=True usually?)
        # Let's check Entity default.
        # Actually Agent inherits Entity.
        # We should check if they block each other.
        
        env = MockEnv(world, [a1, a2], simultaneous_moves=True)
        
        env.take_turn()
        
        # Expectation: They block each other because the target square is occupied at the moment of move()
        self.assertEqual(a1.location, (0,0,0))
        self.assertEqual(a2.location, (0,1,0))

if __name__ == '__main__':
    unittest.main()
