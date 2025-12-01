"""Custom StagHunt environment with integrated probe testing.

This module extends the base StagHuntEnv to include probe testing functionality
during the main training loop.
"""

import os
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
        
        for epoch in range(self.config.experiment.epochs + 1):
            # Reset the environment at the start of each epoch
            self.reset()

            # Determine whether to animate this turn (reuse existing logic)
            animate_this_turn = animate and (
                epoch % self.config.experiment.record_period == 0
            )

            # Start epoch action for each agent model (only spawned agents)
            for agent in self.agents:
                if agent.agent_id in self.spawned_agent_ids:
                    agent.model.start_epoch_action(epoch=epoch)

            # Run the environment for the specified number of turns (reuse existing logic)
            while not self.turn >= self.config.experiment.max_turns:
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
                run_probe_test(self, epoch, output_dir)
            
            # Save models if enabled and it's time (new functionality)
            if (self.config.experiment.get("save_models", False) and 
                epoch > 0 and epoch % self.config.experiment.get("save_interval", 1000) == 0):
                self._save_agent_models(epoch, output_dir)
    
    def _save_agent_models(self, epoch: int, output_dir: Path | None) -> None:
        """Save agent models to disk.
        
        Args:
            epoch: Current epoch number
            output_dir: Directory to save models to
        """
        if output_dir is None:
            output_dir = Path(os.getcwd()) / "./data/"
        
        # Create models directory
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Get experiment name and timestamp for file naming
        experiment_name = self.config.experiment.get("run_name", "experiment")
        timestamp = getattr(self, 'timestamp', 'unknown')
        
        print(f"Saving agent models at epoch {epoch}")
        
        # Save each agent's model
        for agent_id, agent in enumerate(self.agents):
            # Create filename: experiment_name_timestamp_agent_X_epoch_Y.pth
            model_filename = f"{experiment_name}_{timestamp}_agent_{agent_id}_epoch_{epoch}.pth"
            model_path = models_dir / model_filename
            
            try:
                agent.model.save(model_path)
                print(f"  Saved agent {agent_id} model to {model_path}")
            except Exception as e:
                print(f"  Failed to save agent {agent_id} model: {e}")
        
        print(f"Model saving completed for epoch {epoch}")