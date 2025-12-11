"""Probe test runner for executing probe tests and saving results.

This module handles the execution of probe tests during training and saves results to
CSV files for analysis.
"""

import csv
from pathlib import Path

from sorrel.examples.staghunt_physical.probe_test import (
    ProbeTestAgent,
    ProbeTestEnvironment,
)


def run_probe_test(experiment_env, epoch, output_dir):
    """Run probe test and save results to CSV.

    Args:
        experiment_env: The main training environment
        epoch: Current epoch number
        output_dir: Directory to save results
    """
    config = experiment_env.config
    probe_config = config.get("probe_test", {})

    if not probe_config.get("enabled", False):
        return

    # Check if using test_intention mode
    test_mode = probe_config.get("test_mode", "default")

    if test_mode == "test_intention":
        from sorrel.examples.staghunt_physical.probe_test import TestIntentionProbeTest

        test_intention = TestIntentionProbeTest(
            experiment_env, probe_config, output_dir
        )
        test_intention.run_test_intention(experiment_env.agents, epoch)
        print(f"Test intention probe test completed for epoch {epoch}")
        return

    test_epochs = probe_config.get("test_epochs", 1)
    print(f"Running probe test at epoch {epoch} with {test_epochs} test epochs")

    # Create probe test environment (reuse existing classes)
    probe_env = ProbeTestEnvironment(experiment_env, probe_config)

    # Create unit test directory
    unit_test_dir = output_dir / "unit_test"
    unit_test_dir.mkdir(parents=True, exist_ok=True)

    if probe_config.get("individual_testing", True):
        _run_individual_tests(
            experiment_env, probe_env, epoch, unit_test_dir, test_epochs
        )
    else:
        _run_group_test(experiment_env, probe_env, epoch, unit_test_dir, test_epochs)

    print(f"Probe test completed for epoch {epoch}")


def _run_individual_tests(experiment_env, probe_env, epoch, unit_test_dir, test_epochs):
    """Run individual agent tests and save results.

    Args:
        experiment_env: The main training environment
        probe_env: The probe test environment
        epoch: Current epoch number
        unit_test_dir: Directory to save results
        test_epochs: Number of test epochs to run
    """
    # Test each agent individually
    for agent_id, original_agent in enumerate(experiment_env.agents):
        print(f"  Testing agent {agent_id} individually ({test_epochs} epochs)")

        # Create frozen copy of agent
        probe_agent = ProbeTestAgent(original_agent)

        # Run multiple test epochs and save each separately
        csv_filename = f"probe_test_epoch_{epoch}_agent_{agent_id}.csv"
        csv_path = unit_test_dir / csv_filename

        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = _get_csv_fieldnames()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for test_epoch in range(test_epochs):
                agent_metrics = probe_env.run_individual_test(probe_agent, agent_id)
                writer.writerow(
                    _create_csv_row(
                        epoch, agent_id, "individual", agent_metrics, test_epoch
                    )
                )

        print(f"    Results saved to {csv_path}")


def _run_group_test(experiment_env, probe_env, epoch, unit_test_dir, test_epochs):
    """Run group test and save results.

    Args:
        experiment_env: The main training environment
        probe_env: The probe test environment
        epoch: Current epoch number
        unit_test_dir: Directory to save results
        test_epochs: Number of test epochs to run
    """
    print(f"  Testing all agents together ({test_epochs} epochs)")

    # Create frozen copies of all agents
    probe_agents = [ProbeTestAgent(agent) for agent in experiment_env.agents]

    # Save results to CSV with each test epoch as a separate row
    csv_filename = f"probe_test_epoch_{epoch}_group.csv"
    csv_path = unit_test_dir / csv_filename

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = _get_csv_fieldnames()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Run multiple test epochs and save each separately
        for test_epoch in range(test_epochs):
            epoch_metrics = probe_env.run_group_test(probe_agents)

            # Write a row for each agent in this test epoch
            for agent_id, agent_metrics in epoch_metrics.items():
                writer.writerow(
                    _create_csv_row(epoch, agent_id, "group", agent_metrics, test_epoch)
                )

    print(f"    Results saved to {csv_path}")


def _aggregate_metrics(metrics_list):
    """Aggregate metrics across multiple test epochs.

    Args:
        metrics_list: List of metric dictionaries from multiple test epochs

    Returns:
        Dictionary with aggregated metrics (mean values)
    """
    if not metrics_list:
        return {}

    # Initialize aggregated metrics with zeros
    aggregated = {}
    for key in metrics_list[0].keys():
        if isinstance(metrics_list[0][key], dict):
            # Handle defaultdict objects by converting to regular dict
            aggregated[key] = {}
        else:
            aggregated[key] = 0.0

    # Sum all metrics
    for metrics in metrics_list:
        for key, value in metrics.items():
            if isinstance(value, dict):
                # Handle defaultdict objects
                if key not in aggregated:
                    aggregated[key] = {}
                # Convert defaultdict to regular dict and merge
                if hasattr(value, "default_factory"):
                    value = dict(value)
                for sub_key, sub_value in value.items():
                    if sub_key not in aggregated[key]:
                        aggregated[key][sub_key] = 0.0
                    aggregated[key][sub_key] += sub_value
            else:
                # Handle regular numeric values
                aggregated[key] += value

    # Calculate means
    num_epochs = len(metrics_list)
    for key in aggregated.keys():
        if isinstance(aggregated[key], dict):
            # For dict values, calculate mean for each sub-key
            for sub_key in aggregated[key].keys():
                aggregated[key][sub_key] = aggregated[key][sub_key] / num_epochs
        else:
            # For numeric values, calculate mean
            aggregated[key] = aggregated[key] / num_epochs

    return aggregated


def _save_agent_metrics_to_csv(
    csv_path, epoch, agent_id, test_type, agent_metrics, test_epoch=0
):
    """Save agent metrics to CSV file.

    Args:
        csv_path: Path to CSV file
        epoch: Current epoch number
        agent_id: Agent ID
        test_type: Type of test ('individual' or 'group')
        agent_metrics: Dictionary of agent metrics
        test_epoch: Test epoch number (0-based)
    """
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = _get_csv_fieldnames()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            _create_csv_row(epoch, agent_id, test_type, agent_metrics, test_epoch)
        )


def _get_csv_fieldnames():
    """Get the fieldnames for CSV files.

    Returns:
        List of fieldnames for CSV files
    """
    return [
        "epoch",
        "agent_id",
        "test_type",
        "test_epoch",
        "attacks_to_hares",
        "attacks_to_stags",
        "punishments_given",
        "punishments_received",
        "total_reward",
        "attack_cost_paid",
        "punish_cost_paid",
        "resources_defeated",
        "stags_defeated",
        "hares_defeated",
        "shared_rewards_received",
    ]


def _create_csv_row(epoch, agent_id, test_type, agent_metrics, test_epoch=0):
    """Create a CSV row from agent metrics.

    Args:
        epoch: Current epoch number
        agent_id: Agent ID
        test_type: Type of test ('individual' or 'group')
        agent_metrics: Dictionary of agent metrics
        test_epoch: Test epoch number (0-based)

    Returns:
        Dictionary representing a CSV row
    """
    return {
        "epoch": epoch,
        "agent_id": agent_id,
        "test_type": test_type,
        "test_epoch": test_epoch,
        "attacks_to_hares": agent_metrics["attacks_to_hares"],
        "attacks_to_stags": agent_metrics["attacks_to_stags"],
        "punishments_given": agent_metrics["punishments_given"],
        "punishments_received": agent_metrics["punishments_received"],
        "total_reward": agent_metrics["total_reward"],
        "attack_cost_paid": agent_metrics["attack_cost_paid"],
        "punish_cost_paid": agent_metrics["punish_cost_paid"],
        "resources_defeated": agent_metrics["resources_defeated"],
        "stags_defeated": agent_metrics["stags_defeated"],
        "hares_defeated": agent_metrics["hares_defeated"],
        "shared_rewards_received": agent_metrics["shared_rewards_received"],
    }
