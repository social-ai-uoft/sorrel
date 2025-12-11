"""Actor process for environment interaction."""

import time
from pathlib import Path

import numpy as np
from numpy import ndenumerate

from sorrel.agents import Agent
from sorrel.entities import Entity
from sorrel.examples.treasurehunt_mp.mp.mp_shared_models import get_published_policy
from sorrel.utils.logging import ConsoleLogger, Logger
from sorrel.utils.visualization import ImageRenderer


class ActorProcess:
    """Actor process that runs the environment.

    This process steps the environment, queries published policies, and writes
    experiences to shared replay buffers.
    """

    def __init__(
        self,
        env,
        agents,
        shared_state,
        shared_buffers,
        shared_models,
        config,
        logger_queue=None,
    ):
        """Initialize actor process.

        Args:
            env: Environment instance
            agents: List of agents
            shared_state: Shared state dictionary
            shared_buffers: List of shared replay buffers (one per agent)
            shared_models: Shared models (format depends on publish_mode)
            config: MPConfig object
            logger_queue: Queue for sending metrics to main process (optional)
        """
        self.env = env
        self.agents = agents
        self.shared_state = shared_state
        self.shared_buffers = shared_buffers
        self.shared_models = shared_models
        self.config = config
        self.publish_mode = config.publish_mode
        self.logger_queue = logger_queue

        # Animation setup
        self.renderer = None
        self.animate = config.logging  # Use logging flag to determine animation

    def run(self):
        """Main actor loop."""
        try:
            # Setup animation if enabled
            if self.animate:
                self.renderer = ImageRenderer(
                    experiment_name=self.env.world.__class__.__name__,
                    record_period=self.config.record_period,
                    num_turns=self.config.max_turns,
                )

            # Main epoch loop
            for epoch in range(self.config.epochs + 1):
                if self.shared_state["should_stop"].value:
                    break

                # Print progress
                if epoch % max(1, self.config.epochs // 10) == 0 or epoch == 0:
                    print(f"Actor: Epoch {epoch}/{self.config.epochs}")

                # Reset environment at start of each epoch
                self.env.reset()

                # Determine whether to animate this epoch
                animate_this_epoch = self.animate and (
                    epoch % self.config.record_period == 0
                )

                # Start epoch action for each agent model
                for agent in self.agents:
                    # Use published model for start_epoch_action
                    # Note: We need to handle this carefully since models are shared
                    pass  # Skip for now, can be added if needed

                # Run environment for specified number of turns
                self.env.turn = 0
                while self.env.turn < self.config.max_turns:
                    if self.shared_state["should_stop"].value:
                        break

                    # Render if needed
                    if animate_this_epoch and self.renderer is not None:
                        self.renderer.add_image(self.env.world)

                    # Step environment
                    self.step_environment()

                    # Increment turn counter (local to actor process)
                    self.env.turn += 1

                    # Increment global epoch counter (represents turns/experience collection)
                    with self.shared_state["global_epoch"].get_lock():
                        self.shared_state["global_epoch"].value += 1

                self.env.world.is_done = True

                # Collect metrics for logging
                total_reward = self.env.world.total_reward
                total_loss = 0.0
                epsilon = 0.0

                # Get epsilon from published model (use first agent's epsilon as representative)
                if len(self.agents) > 0:
                    published_model = get_published_policy(
                        0, self.shared_models, self.shared_state, self.config
                    )
                    epsilon = getattr(published_model, "epsilon", 0.0)

                # Aggregate losses from shared state (learner processes write here)
                agent_losses_dict = {}
                for i in range(len(self.agents)):
                    if "agent_losses" in self.shared_state:
                        with self.shared_state["agent_loss_counts"][i].get_lock():
                            count = self.shared_state["agent_loss_counts"][i].value
                        if count > 0:
                            # Get recent losses from shared array (last 100 training steps)
                            losses_array = self.shared_state["agent_losses"][i]
                            # Use circular buffer: get last min(count, 100) losses
                            start_idx = max(0, count - 100)
                            recent_losses = [
                                losses_array[j % 100] for j in range(start_idx, count)
                            ]
                            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
                            agent_losses_dict[f"agent_{i}_loss"] = avg_loss
                            total_loss += avg_loss

                # Send metrics to main process for logging
                if self.logger_queue is not None:
                    metrics = {
                        "epoch": epoch,
                        "total_reward": total_reward,
                        "total_loss": (
                            total_loss / len(self.agents)
                            if len(self.agents) > 0
                            else 0.0
                        ),
                        "epsilon": epsilon,
                        **agent_losses_dict,
                    }
                    try:
                        self.logger_queue.put(metrics, block=False)
                    except:
                        pass  # Queue full, skip this epoch's logging

                # Print epoch completion
                if (
                    epoch % max(1, self.config.epochs // 10) == 0
                    or epoch == self.config.epochs
                ):
                    print(
                        f"Actor: Epoch {epoch} completed. Total reward: {total_reward:.2f}"
                    )

                # Generate GIF if animation was done
                if animate_this_epoch and self.renderer is not None:
                    output_dir = Path(self.config.log_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    self.renderer.save_gif(epoch, output_dir)

                # End epoch action
                for agent in self.agents:
                    pass  # Skip for now, can be added if needed

                # Check for termination
                if self.env.world.is_done:
                    break

        except KeyboardInterrupt:
            print("Actor process interrupted")
        except Exception as e:
            print(f"Actor process error: {e}")
            import traceback

            traceback.print_exc()
            self.shared_state["actor_error_flag"].value = True
        finally:
            self.cleanup()

    def step_environment(self):
        """Single environment step - uses sequential agent transitions like original code.

        This follows the original Environment.take_turn() logic:
        1. Transition non-agent entities first
        2. Transition each agent sequentially (to avoid conflicts)
        """
        # 1. Transition non-agent entities first
        for _, x in ndenumerate(self.env.world.map):
            x: Entity
            if x.has_transitions and not isinstance(x, Agent):
                x.transition(self.env.world)

        # 2. Transition each agent sequentially (to avoid conflicts)
        # This follows the original agent.transition() logic but uses:
        # - Published model for action selection
        # - Shared buffer for experience storage
        for i, agent in enumerate(self.agents):
            # Get observation (same as agent.pov())
            state = agent.pov(self.env.world)

            # Get published policy for this agent
            published_model = get_published_policy(
                i, self.shared_models, self.shared_state, self.config
            )

            # Get action using published model
            # We need to use the shared buffer's current_state() for frame stacking
            # Temporarily replace agent's model for action selection
            original_model = agent.model
            agent.model = published_model

            # Replace agent's memory with shared buffer for state stacking
            original_memory = agent.model.memory
            agent.model.memory = self.shared_buffers[i]

            # Get action (this will use published_model with shared_buffer for state stacking)
            action = agent.get_action(state)

            # Execute action (this updates the world)
            reward = agent.act(self.env.world, action)
            done = agent.is_done(self.env.world)

            # Restore original model and memory
            agent.model.memory = original_memory
            agent.model = original_model

            # Store experience in shared buffer (agent.add_memory would use agent.model.memory,
            # so we manually add to shared buffer)
            self.shared_buffers[i].add(
                obs=state, action=action, reward=reward, done=done
            )

            # Update world total reward (normally done in agent.transition)
            self.env.world.total_reward += reward

    def cleanup(self):
        """Clean up resources."""
        if self.renderer is not None:
            # Cleanup renderer if needed
            pass
