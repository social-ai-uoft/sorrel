"""Custom StagHunt environment with integrated probe testing.

This module extends the base StagHuntEnv to include probe testing functionality
during the main training loop.
"""

import os
import random
from pathlib import Path

from sorrel.utils.logging import Logger

from sorrel.examples.staghunt_physical.env import StagHuntEnv
from sorrel.examples.staghunt_physical.probe_test_runner import run_probe_test


class StagHuntEnvWithProbeTest(StagHuntEnv):
    """StagHuntEnv with integrated probe testing."""
    
    def run_experiment(
        self,
        animate: bool = True,
        logging: bool = True,
        logger: Logger | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Run experiment with integrated probe testing.
        
        This method extends the base run_experiment to include probe testing
        at specified intervals during training.
        """
        from sorrel.utils.visualization import ImageRenderer
        
        # Initialize renderer if animation is enabled (reuse existing logic)
        renderer = None
        if animate:
            renderer = ImageRenderer(
                experiment_name=self.world.__class__.__name__,
                record_period=self.config.experiment.record_period,
                num_turns=self.config.experiment.max_turns,
            )
        
        # Check if probe testing is enabled
        probe_config = self.config.get("probe_test", {})
        probe_enabled = probe_config.get("enabled", False)
        test_interval = probe_config.get("test_interval", 1000)
        
        # Track if first probe test export has been done (initialize on first call)
        if not hasattr(self, 'first_probe_test_exported'):
            self.first_probe_test_exported = False
        
        for epoch in range(self.config.experiment.epochs + 1):
            # Reset the environment at the start of each epoch
            self.reset()

            # Sample random max turns if random_max_turns is enabled
            random_max_turns = self.config.experiment.get("random_max_turns", False)
            if random_max_turns:
                max_turns = self.config.experiment.max_turns
                # Sample from [1, max_turns] inclusive
                self.current_epoch_max_turns = random.randint(1, max_turns)
            else:
                # Use fixed max_turns
                self.current_epoch_max_turns = self.config.experiment.max_turns

            # Determine whether to animate this turn (reuse existing logic)
            animate_this_turn = animate and (
                epoch % self.config.experiment.record_period == 0
            )

            # Start epoch action for each agent model (only spawned agents)
            for agent in self.agents:
                if agent.agent_id in self.spawned_agent_ids:
                    agent.model.start_epoch_action(epoch=epoch)

            # Run the environment for the specified number of turns
            # Use current_epoch_max_turns which may be randomly sampled
            while not self.turn >= self.current_epoch_max_turns:
                if animate_this_turn and renderer is not None:
                    renderer.add_image(self.world)
                self.take_turn()

            self.world.is_done = True

            # Generate the gif if animation was done (reuse existing logic)
            if animate_this_turn and renderer is not None:
                if output_dir is None:
                    output_dir = Path(os.getcwd()) / "./data/"
                renderer.save_gif(epoch, output_dir)

            # End epoch action for each agent model (only spawned agents)
            for agent in self.agents:
                if agent.agent_id in self.spawned_agent_ids:
                    agent.model.end_epoch_action(epoch=epoch)

            # Train the agents (only spawned agents)
            total_loss = 0
            for agent in self.agents:
                if agent.agent_id in self.spawned_agent_ids:
                    total_loss += agent.model.train_step()

            # Update epsilon for all agents (only spawned agents)
            for agent in self.agents:
                if agent.agent_id in self.spawned_agent_ids:
                    agent.model.epsilon_decay(self.config.model.epsilon_decay)

            # Log the information (reuse existing logic)
            if logging:
                if not logger:
                    from sorrel.utils.logging import ConsoleLogger
                    logger = ConsoleLogger(self.config.experiment.epochs)
                logger.record_turn(
                    epoch,
                    total_loss,
                    self.world.total_reward,
                    self.agents[0].model.epsilon,
                )

            # Run probe test if enabled and it's time (new functionality)
            if probe_enabled and epoch > 0 and epoch % test_interval == 0:
                if output_dir is None:
                    output_dir = Path(os.getcwd()) / "./data/"
                
                # Export identity codes on first probe test (before running probe test)
                if not self.first_probe_test_exported:
                    if hasattr(self, 'agents') and self.agents and len(self.agents) > 0:
                        try:
                            from sorrel.examples.staghunt_physical.env import export_agent_identity_codes
                            export_agent_identity_codes(
                                agents=self.agents,
                                output_dir=output_dir,
                                epoch=epoch,
                                context="probe_test"
                            )
                        except ImportError:
                            # Function not available (shouldn't happen, but handle gracefully)
                            print(f"Warning: Could not import export_agent_identity_codes function")
                        except Exception as e:
                            # Any other error during export (log but don't stop probe test)
                            print(f"Warning: Error exporting identity codes: {e}")
                    self.first_probe_test_exported = True
                
                # Run probe test (existing functionality, unchanged)
                run_probe_test(self, epoch, output_dir)
            
            # Save models if enabled and it's time (new functionality)
            if (self.config.experiment.get("save_models", False) and 
                epoch > 0 and epoch % self.config.experiment.get("save_interval", 1000) == 0):
                self._save_agent_models(epoch, output_dir)
    
    def _save_agent_models(self, epoch: int, output_dir: Path | None) -> None:
        """Save agent models to disk.
        
        Models are saved to a 'models' folder at the root of staghunt_physical.
        Each agent has one model file that gets overwritten on each save (no epoch in filename).
        
        Args:
            epoch: Current epoch number (for logging purposes)
            output_dir: Not used, kept for compatibility
        """
        # Create models directory at the root of staghunt_physical
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Get experiment name for file naming
        experiment_name = self.config.experiment.get("run_name", "experiment")
        
        print(f"Saving agent models at epoch {epoch}")
        
        # Save each agent's model (overwrite previous versions)
        for agent_id, agent in enumerate(self.agents):
            # Create filename without epoch number so it overwrites old checkpoints
            model_filename = f"{experiment_name}_agent_{agent_id}.pth"
            model_path = models_dir / model_filename
            
            try:
                agent.model.save(model_path)
                print(f"  Saved agent {agent_id} model to {model_path}")
            except Exception as e:
                print(f"  Failed to save agent {agent_id} model: {e}")
        
        print(f"Model saving completed for epoch {epoch}")