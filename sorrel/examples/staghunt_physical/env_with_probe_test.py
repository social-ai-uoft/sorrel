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

            # Start epoch action for each agent model (reuse existing logic)
            for agent in self.agents:
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

            # End epoch action for each agent model (reuse existing logic)
            for agent in self.agents:
                agent.model.end_epoch_action(epoch=epoch)

            # Train the agents (reuse existing logic)
            total_loss = 0
            for agent in self.agents:
                total_loss += agent.model.train_step()

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