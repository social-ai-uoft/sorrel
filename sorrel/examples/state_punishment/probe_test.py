"""Probe test implementation for state punishment experiments."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

from sorrel.examples.state_punishment.environment_setup import setup_environments
from sorrel.utils.logging import TensorboardLogger

# Hardcoded probe test configuration
PROBE_TEST_CONFIG = {
    "frequency": 0,  # Set to 0 to disable probe tests, >0 to enable
    "epochs": 10,  # Number of probe test epochs per test
    "epsilon": 0.0,  # Epsilon for probe test (0 = no exploration)
    "freeze_networks": True,  # Freeze neural networks during probe test
    "save_models": True,  # Save model checkpoints during probe tests
    "use_important_rule": None,  # None=inherit from training, True=force important, False=force silly
    "use_fixed_seed": True,  # Use fixed seed for reproducible spatial layout
    "fixed_seed": 42,  # Fixed seed for probe test environment
}


class ProbeTestLogger:
    """Separate logger for probe test results - uses same metrics as training."""

    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.probe_test_logger = TensorboardLogger(10000, log_dir / "probe_tests")
        self.probe_test_results = []

    def record_probe_test(self, training_epoch: int, probe_results: dict):
        """Record probe test results - each probe test is one timestep."""
        # Add probe_test prefix to all metric names
        prefixed_metrics = {}
        for key, value in probe_results["metrics"].items():
            prefixed_metrics[f"probe_test_{key}"] = value

        # Also prefix the main reward metric
        prefixed_metrics["probe_test_avg_total_reward"] = probe_results[
            "avg_total_reward"
        ]

        # Log to tensorboard with probe_test prefix
        self.probe_test_logger.record_turn(
            training_epoch,  # Use training epoch as the "epoch" for probe tests
            0.0,  # No loss during probe test
            probe_results[
                "avg_total_reward"
            ],  # Keep original for backward compatibility
            0.0,  # Epsilon = 0 for probe tests
            **prefixed_metrics,  # All metrics with probe_test prefix
        )

        # Store results
        self.probe_test_results.append(
            {
                "training_epoch": training_epoch,
                "probe_results": probe_results,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def save_probe_test_results(self):
        """Save probe test results to file."""
        results_file = self.log_dir / "probe_tests" / "probe_test_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(self.probe_test_results, f, indent=2)


def setup_probe_test_environment(config, args, use_important_rule=None):
    """Create probe test environment - identical to training except for rule choice and fixed seed."""
    import numpy as np
    import torch

    # Copy config to avoid modifying original
    probe_config = OmegaConf.create(OmegaConf.to_yaml(config))

    # Override important_rule if specified
    if use_important_rule is not None:
        probe_config.experiment.important_rule = use_important_rule

    # Store original random states
    original_numpy_state = np.random.get_state()
    original_torch_state = torch.get_rng_state()

    # Set fixed seed for reproducible spatial layout if enabled
    if PROBE_TEST_CONFIG["use_fixed_seed"]:
        np.random.seed(PROBE_TEST_CONFIG["fixed_seed"])
        torch.manual_seed(PROBE_TEST_CONFIG["fixed_seed"])

    # Create probe test environment using same setup as training
    probe_multi_agent_env, probe_shared_state_system, probe_shared_social_harm = (
        setup_environments(
            probe_config,
            args.simple_foraging,
            args.fixed_punishment,
            args.random_policy,
            run_folder=f"{args.run_folder}_probe_test",
        )
    )

    # Restore original random states to not affect training
    if PROBE_TEST_CONFIG["use_fixed_seed"]:
        np.random.set_state(original_numpy_state)
        torch.set_rng_state(original_torch_state)

    return probe_multi_agent_env, probe_shared_state_system, probe_shared_social_harm


def copy_and_freeze_model_weights(source_model, target_model):
    """Copy neural network weights and freeze the target model."""
    # Copy local network weights
    target_model.qnetwork_local.load_state_dict(
        source_model.qnetwork_local.state_dict()
    )
    # Copy target network weights
    target_model.qnetwork_target.load_state_dict(
        source_model.qnetwork_target.state_dict()
    )

    # Freeze the networks
    for param in target_model.qnetwork_local.parameters():
        param.requires_grad = False
    for param in target_model.qnetwork_target.parameters():
        param.requires_grad = False

    # Set epsilon to 0 for probe test
    target_model.epsilon = 0.0


def unfreeze_model_weights(model):
    """Unfreeze model weights after probe test."""
    for param in model.qnetwork_local.parameters():
        param.requires_grad = True
    for param in model.qnetwork_target.parameters():
        param.requires_grad = True


def aggregate_metrics_across_epochs(metrics_list):
    """Aggregate metrics across all probe epochs."""
    if not metrics_list:
        return {}

    # Get all metric keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    # Calculate averages for each metric
    aggregated = {}
    for key in all_keys:
        values = [metrics.get(key, 0) for metrics in metrics_list]
        aggregated[key] = np.mean(values)

    return aggregated


def run_probe_test(
    training_env, probe_env, training_epoch: int, probe_epochs: int = 10
):
    """Run probe test - structurally identical to training but with frozen models."""
    import numpy as np
    import torch

    # Store original random states
    original_numpy_state = np.random.get_state()
    original_torch_state = torch.get_rng_state()

    # Set fixed seed for reproducible probe test execution if enabled
    if PROBE_TEST_CONFIG["use_fixed_seed"]:
        np.random.seed(PROBE_TEST_CONFIG["fixed_seed"])
        torch.manual_seed(PROBE_TEST_CONFIG["fixed_seed"])

    # Copy and freeze model weights from training to probe agents
    for train_env, probe_env in zip(
        training_env.individual_envs, probe_env.individual_envs
    ):
        train_agent = train_env.agents[0]
        probe_agent = probe_env.agents[0]
        copy_and_freeze_model_weights(train_agent.model, probe_agent.model)

    # Run probe test epochs - use same structure as training
    probe_results = {
        "total_rewards": [],
        "metrics": [],  # Will collect same metrics as training
    }

    for probe_epoch in range(probe_epochs):
        # Reset probe environment
        probe_env.reset()

        # Start epoch action for all agents
        for env in probe_env.individual_envs:
            for agent in env.agents:
                agent.model.start_epoch_action(epoch=probe_epoch)

        # Run the environment for the specified number of turns
        total_reward = 0.0
        while not probe_env.turn >= probe_env.config.experiment.max_turns:
            probe_env.take_turn()
            total_reward += probe_env.world.total_reward

            # Check if any environment is done
            if any(env.world.is_done for env in probe_env.individual_envs):
                break

        # Set all environments as done
        for env in probe_env.individual_envs:
            env.world.is_done = True

        # End epoch action for all agents
        for env in probe_env.individual_envs:
            for agent in env.agents:
                agent.model.end_epoch_action(epoch=probe_epoch)

        # Collect results using the same metrics as training
        probe_results["total_rewards"].append(total_reward)

        # Get metrics using the same method as training
        epoch_metrics = {}
        for i, env in enumerate(probe_env.individual_envs):
            agent = env.agents[0]

            # Individual agent metrics (same as training)
            epoch_metrics[f"Agent_{i}/individual_score"] = agent.individual_score

            # Encounter metrics (same as training)
            for entity_type, count in agent.encounters.items():
                epoch_metrics[f"Agent_{i}/{entity_type}_encounters"] = count

            # Action frequency metrics (same as training)
            for action_name, frequency in agent.action_frequencies.items():
                epoch_metrics[f"Agent_{i}/action_freq_{action_name}"] = frequency

            # Sigma weights (same as training)
            if hasattr(agent.model, "qnetwork_local") and hasattr(
                agent.model.qnetwork_local, "ff_1"
            ):
                epoch_metrics[f"Agent_{i}/sigma_weight_ff1"] = (
                    agent.model.qnetwork_local.ff_1.sigma_weight.mean().item()
                )
                epoch_metrics[f"Agent_{i}/sigma_weight_advantage"] = (
                    agent.model.qnetwork_local.advantage.sigma_weight.mean().item()
                )
                epoch_metrics[f"Agent_{i}/sigma_weight_value"] = (
                    agent.model.qnetwork_local.value.sigma_weight.mean().item()
                )

        # Global metrics (same as training)
        epoch_metrics["Global/average_punishment_level"] = (
            probe_env.shared_state_system.prob
        )
        epoch_metrics["Global/current_punishment_level"] = (
            probe_env.shared_state_system.prob
        )

        probe_results["metrics"].append(epoch_metrics)

    # Calculate aggregate results (averages across all probe epochs)
    aggregate_results = {
        "avg_total_reward": np.mean(probe_results["total_rewards"]),
        "std_total_reward": np.std(probe_results["total_rewards"]),
        "metrics": aggregate_metrics_across_epochs(probe_results["metrics"]),
        "additional_info": {
            "probe_epochs": probe_epochs,
            "training_epoch": training_epoch,
            "epsilon": 0.0,
            "frozen_networks": True,
        },
    }

    # Unfreeze models after probe test
    for env in probe_env.individual_envs:
        agent = env.agents[0]
        unfreeze_model_weights(agent.model)

    # Restore original random states to not affect training
    if PROBE_TEST_CONFIG["use_fixed_seed"]:
        np.random.set_state(original_numpy_state)
        torch.set_rng_state(original_torch_state)

    return aggregate_results


def save_probe_test_models(probe_env, epoch: int, experiment_name: str):
    """Save probe test models using existing model saving infrastructure."""
    # Create probe test models directory
    models_dir = Path(__file__).parent / "models" / "probe_tests"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save each agent's model
    for env_idx, env in enumerate(probe_env.individual_envs):
        for agent_idx, agent in enumerate(env.agents):
            model_filename = f"{experiment_name}_probe_test_epoch_{epoch}_env_{env_idx}_agent_{agent_idx}.pth"
            model_path = models_dir / model_filename

            # Save the model
            agent.model.save(model_path)

    print(f"Saved probe test models for epoch {epoch} to {models_dir.absolute()}")
