"""Entities for the Stag Hunt environment implemented in the Sorrel framework.

These classes derive from :class:`sorrel.entities.Entity` and describe the
different types of objects that can occupy cells in the grid.  Entities
encapsulate appearance (sprites), passability and any intrinsic reward when
encountered.  Resources correspond to the ``stag`` and ``hare`` pickups in
the classic stag‑hunt social dilemma.  Walls are impassable.  Empty cells
allow movement and can transition into resources via the regeneration logic
implemented in the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sorrel.entities import Entity

if TYPE_CHECKING:
    from sorrel.examples.staghunt_physical.world import StagHuntWorld


entity_list = [
            "Empty",
            "Wall",
            "Spawn",
            "StagResource",
            "WoundedStagResource",  # Wounded stag type (kind changes when health < max_health)
            "HareResource",
            "StagHuntAgentNorth",  # 0: north
            "StagHuntAgentEast",  # 1: east
            "StagHuntAgentSouth",  # 2: south
            "StagHuntAgentWest",  # 3: west
            "Sand",
            "AttackBeam",
            "PunishBeam",
        ]


class Wall(Entity["StagHuntWorld"]):
    """An impassable wall entity.

    Walls block agent movement and zapping beams.  They carry no intrinsic reward and
    never change state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.passable = False
        # negative reward when agents bump into walls (optional)
        self.value = 0
        # Path to wall sprite; placeholder image filename
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class Sand(Entity["StagHuntWorld"]):
    """An entity that represents a block of sand in the stag hunt environment.

    Sand entities track whether they can spawn resources and manage respawn timing. They
    also remember what type of resource should respawn at this location.
    """

    def __init__(
        self,
        can_convert_to_resource: bool = False,
        respawn_ready: bool = True,
        resource_type: str = None,
    ):
        super().__init__()
        # We technically don't need to make Sand passable here since it's on a different layer from Agent
        self.passable = True
        self.can_convert_to_resource = can_convert_to_resource
        self.respawn_ready = respawn_ready
        self.respawn_timer = 0  # Timer for respawn lag
        self.has_transitions = True  # Enable transitions for respawn timing
        self.resource_type = resource_type  # 'stag', 'hare', or None for random
        self.sprite = Path(__file__).parent / "./assets/sand.png"

    def transition(self, world: StagHuntWorld) -> None:
        """Handle respawn timing for resource spawn points.

        Sand entities that can convert to resources but are not ready will count down
        their respawn timer until they become ready again.
        """
        if self.can_convert_to_resource and not self.respawn_ready:
            self.respawn_timer += 1
            if self.respawn_timer >= world.respawn_lag:
                self.respawn_ready = True
                self.respawn_timer = 0


class Empty(Entity["StagHuntWorld"]):
    """An empty traversable cell.

    Empty cells hold no resources and allow agents to move through.  In the
    regeneration step, empty cells can spawn new resources with probability
    determined by the world's ``resource_density`` hyperparameter. The ability
    to spawn resources is inherited from the terrain layer below.
    """

    def __init__(self) -> None:
        super().__init__()
        self.passable = True
        self.value = 0
        self.sprite = Path(__file__).parent / "./assets/empty.png"

    def transition(self, world: StagHuntWorld) -> None:
        """Randomly spawn a resource on this cell during regeneration.

        When the world performs its regeneration step, empty cells may spawn
        either a StagResource or HareResource, but only if the terrain below
        can convert to a resource and is ready to respawn. The probability is
        given by ``world.resource_density`` and the class is selected based on
        the resource type remembered by the Sand entity below.  If a resource
        is spawned, the empty entity is replaced.
        """
        # Check the terrain layer below for resource spawn capability and readiness
        terrain_location = (self.location[0], self.location[1], world.terrain_layer)
        if world.valid_location(terrain_location):
            terrain_entity = world.observe(terrain_location)
            
            # Check if there's an agent at the current location (dynamic layer)
            # We need to check all layers to see if there's an agent above this empty cell
            has_agent = False
            for layer in range(world.layers):
                check_location = (self.location[0], self.location[1], layer)
                if world.valid_location(check_location):
                    entity_at_layer = world.observe(check_location)
                    # Check if it's an agent (any orientation)
                    if hasattr(entity_at_layer, 'kind') and 'StagHuntAgent' in entity_at_layer.kind:
                        has_agent = True
                        break
            
            if (
                hasattr(terrain_entity, "can_convert_to_resource")
                and hasattr(terrain_entity, "respawn_ready")
                and terrain_entity.can_convert_to_resource
                and terrain_entity.respawn_ready
                and not has_agent  # Don't spawn if there's an agent above
                and np.random.random() < world.resource_density
            ):

                # Choose resource type based on what's remembered in the Sand entity
                if (
                    hasattr(terrain_entity, "resource_type")
                    and terrain_entity.resource_type
                ):
                    if terrain_entity.resource_type == "stag":
                        res_cls = StagResource
                    elif terrain_entity.resource_type == "hare":
                        res_cls = HareResource
                    else:
                        # Fallback to random selection for unknown types
                        stag_prob = getattr(world, 'stag_probability', 0.2)
                        res_cls = (
                            StagResource if np.random.random() < stag_prob else HareResource
                        )
                else:
                    # Fallback to original random selection if no resource type is remembered
                    stag_prob = getattr(world, 'stag_probability', 0.2)
                    res_cls = StagResource if np.random.random() < stag_prob else HareResource

                # Step 3: Apply resource-specific spawn success rate filter
                should_spawn = True
                if getattr(world, 'dynamic_resource_density_enabled', False):
                    if res_cls == StagResource:
                        should_spawn = np.random.random() < world.current_stag_rate
                    else:  # HareResource
                        should_spawn = np.random.random() < world.current_hare_rate
                
                # Only spawn if Step 3 filter passes (or feature disabled)
                if should_spawn:
                    # Use separate reward values for stag and hare
                    if res_cls == StagResource:
                        reward_value = world.stag_reward
                        health_value = world.stag_health
                        cooldown_value = world.stag_regeneration_cooldown
                    else:  # HareResource
                        reward_value = world.hare_reward
                        health_value = world.hare_health
                        cooldown_value = world.hare_regeneration_cooldown
                    
                    world.add(
                        self.location, res_cls(reward_value, health_value, regeneration_cooldown=cooldown_value)
                    )
                # else: filtered out by Step 3, don't spawn (Empty entity remains)


class Spawn(Entity["StagHuntWorld"]):
    """Spawn point entity.

    Spawn cells mark potential spawn locations for agents.  They are passable so that
    agents may stand on them.  Spawn cells do not produce resources.
    """

    def __init__(self) -> None:
        super().__init__()
        self.passable = (
            True  # Spawn points should be passable so agents can move over them
        )
        self.value = 0
        self.sprite = Path(__file__).parent / "./assets/spawn.png"


class Resource(Entity["StagHuntWorld"]):
    """Base class for resources.

    Resources are passable but deliver a small intrinsic reward when collected.  They
    have health indicating how many zap hits are required to destroy them.  Specific
    subclasses encode which strategy they represent.
    """

    name: str  # overridden in subclasses

    def __init__(self, taste_reward: float, max_health: int, regeneration_rate: float = 0.1, regeneration_cooldown: int = 1) -> None:
        super().__init__()
        self.passable = False  # Resources are not passable - agents must attack them
        self.max_health = max_health
        self.health = max_health
        self.value = taste_reward
        # self.regeneration_rate = regeneration_rate
        self.regeneration_cooldown = regeneration_cooldown
        self.last_attacked_turn = 0
        self.attack_history: list[int] = []  # Track agent IDs that have successfully damaged this resource
        self.has_transitions = True
        # sprite will be set in subclasses

    def on_attack(self, world: StagHuntWorld, current_turn: int, attacker_id: int | None = None) -> bool:
        """Handle an attack on this resource.

        Reduces health by one and updates last attacked turn. Returns True if resource
        is defeated (health reaches zero), False otherwise.
        
        Args:
            world: The game world
            current_turn: Current turn number
            attacker_id: Optional agent ID that performed the attack. Only added to attack_history
                        if provided and the attack successfully decreases health.
        """
        self.health -= 1
        # Add to attack history ONLY if attacker_id is provided and health actually decreased
        if attacker_id is not None and attacker_id not in self.attack_history:
            self.attack_history.append(attacker_id)
        self.last_attacked_turn = current_turn
        
        if self.health <= 0:
            # Mark the Sand entity below as not ready to respawn and remember this resource type
            terrain_location = (self.location[0], self.location[1], world.terrain_layer)
            if world.valid_location(terrain_location):
                terrain_entity = world.observe(terrain_location)
                if (
                    hasattr(terrain_entity, "can_convert_to_resource")
                    and terrain_entity.can_convert_to_resource
                ):
                    terrain_entity.respawn_ready = False
                    terrain_entity.respawn_timer = 0
                    terrain_entity.resource_type = (
                        self.name
                    )  # Remember the resource type

            # replace with empty cell (attributes inherited from terrain layer)
            world.add(self.location, Empty())
            return True
        return False

    def transition(self, world: StagHuntWorld) -> None:
        """Handle health regeneration over time.
        
        Resources regenerate health when they haven't been attacked recently.
        When health regenerates, the attack history is reset.
        """
        old_health = self.health
        # Only regenerate if not at max health and haven't been attacked recently
        if self.health < self.max_health:
            # Get current turn from world (assuming world has a turn counter)
            current_turn = getattr(world, 'current_turn', 0)
            turns_since_attack = current_turn - self.last_attacked_turn
            
            # Regenerate if enough time has passed since last attack
            if turns_since_attack >= self.regeneration_cooldown:
                self.health = min(self.max_health, self.health + world.health_regeneration_rate)
        
        # Reset attack history when health regenerates (resource "resets")
        if self.health > old_health:
            self.attack_history = []


class StagResource(Resource):
    """Resource representing the 'stag' strategy."""

    name = "stag"

    def __init__(self, taste_reward: float, max_health: int = 12, regeneration_rate: float = 0.1, regeneration_cooldown: int = 1) -> None:
        super().__init__(taste_reward, max_health, regeneration_rate, regeneration_cooldown)
        self.sprite = Path(__file__).parent / "./assets/stag.png"
        # Initialize kind as 'StagResource' (full health)
        self.kind = "StagResource"
    
    def _update_kind(self, world: StagHuntWorld) -> None:
        """Update kind based on current health status (only if enabled in world config)."""
        if not getattr(world, 'use_wounded_stag', False):
            return  # Feature disabled, keep kind as 'StagResource'
        
        if self.health < self.max_health:
            self.kind = "WoundedStagResource"
        else:
            self.kind = "StagResource"
    
    def on_attack(self, world: StagHuntWorld, current_turn: int, attacker_id: int | None = None) -> bool:
        """Handle an attack on this resource.
        
        Updates kind to 'WoundedStagResource' if health < max_health (only if enabled).
        """
        # Call parent to handle health reduction
        defeated = super().on_attack(world, current_turn, attacker_id)
        
        # Update kind based on new health (if not defeated and feature enabled)
        if not defeated:
            self._update_kind(world)
        
        return defeated
    
    def transition(self, world: StagHuntWorld) -> None:
        """Handle health regeneration and kind updates.
        
        Updates kind back to 'StagResource' when health returns to max (only if enabled).
        """
        old_health = self.health
        # Call parent to handle regeneration
        super().transition(world)
        
        # Update kind if health changed (regenerated) and feature enabled
        if self.health != old_health:
            self._update_kind(world)


class HareResource(Resource):
    """Resource representing the 'hare' strategy."""

    name = "hare"

    def __init__(self, taste_reward: float, max_health: int = 3, regeneration_rate: float = 0.1, regeneration_cooldown: int = 1) -> None:
        super().__init__(taste_reward, max_health, regeneration_rate, regeneration_cooldown)
        self.sprite = Path(__file__).parent / "./assets/hare.png"


# --------------------------- #
# region: Beams               #
# --------------------------- #


class Beam(Entity["StagHuntWorld"]):
    """Generic beam class for agent interaction beams."""

    def __init__(self, creator_agent_id: int | None = None):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/beam.png"
        self.turn_counter = 0
        self.has_transitions = True
        self.creator_agent_id = creator_agent_id  # Track which agent created this beam

    def transition(self, world: StagHuntWorld):
        # Beams persist for one full turn, then disappear.
        if self.turn_counter >= 1:
            world.add(self.location, Empty())
        else:
            self.turn_counter += 1


class InteractionBeam(Beam):
    """Beam used for agent interactions in stag hunt."""

    def __init__(self):
        super().__init__()


class AttackBeam(Beam):
    """Beam used for attacking resources in stag hunt."""

    def __init__(self, creator_agent_id: int | None = None):
        super().__init__(creator_agent_id)
        self.sprite = Path(__file__).parent / "./assets/beam.png"  # Could use different sprite


class PunishBeam(Beam):
    """Beam used for punishing agents in stag hunt."""

    def __init__(self, creator_agent_id: int | None = None):
        super().__init__(creator_agent_id)
        self.sprite = Path(__file__).parent / "./assets/zap.png"  # Using zap sprite for punish
