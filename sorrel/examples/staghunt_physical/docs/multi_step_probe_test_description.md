# Multi-Step Probe Test Description

## Overview

This study employed multi-step probe tests to measure agent attack preferences and behavioral choices over multiple turns throughout training. Multi-step probe tests create frozen copies of agents at specific training epochs and evaluate their decision-making in controlled test environments over an extended period without allowing the agents to learn or update their parameters. This approach enables assessment of agent behavioral preferences and attack target selection at different stages of training without interfering with the ongoing learning process.

## Multi-Step Probe Test

The study used multi-step probe test mode, which measures agent attack preferences by observing their actual behavioral choices over multiple turns in controlled scenarios. In this mode, agents are placed in carefully designed test environments with specific spatial arrangements containing both stags and hares positioned to create meaningful choices. Unlike the one-step test intention mode that measures Q-values at initialization, the multi-step test allows agents to move, explore, and make actual attack decisions, revealing their behavioral preferences through observed actions rather than inferred intentions.

## Test Procedure

Probe tests were executed every 100 epochs throughout training. When a probe test was triggered, frozen copies of all agents were created. These frozen copies were identical to the original agents in terms of their learned parameters and decision-making models, but were set to evaluation mode with exploration disabled, ensuring that the tests measured the agents' learned policies rather than their exploratory behavior.

Each agent was tested using the `test_multi_step.txt` map, which contains a focal agent spawn point at the center (row 7) and a fake agent spawn point at the upper position (row 4). The map also contains a stag resource at row 5 (above the focal agent) and a hare resource at row 9 (below the focal agent), creating a clear choice between pursuing the high-value stag or the low-value hare. The test environment is smaller than the training environment (13×13 units) and contains precisely positioned resources that require agents to make directional and target selection choices.

## Social Context Variation

To assess how social context influences agent attack preferences, each agent was tested under three different partner conditions. In the first condition, agents were tested alone with no partner agent present. In the second condition, agents were tested with a fake partner agent of type AgentKindA positioned at the upper spawn point. In the third condition, agents were tested with a fake partner agent of type AgentKindB positioned at the upper spawn point. Fake partner agents were configured to remain stationary and not act during the test, serving as social context cues without interfering with the focal agent's decision-making process.

The fake partner agent at the upper position (row 4) was oriented to face south (toward larger row numbers), directly facing the stag resource at row 5. This orientation ensures that the fake agent provides a clear social signal about the presence of the stag, allowing assessment of how the focal agent's attack preferences change depending on the type of partner agent present and whether the partner appears to be oriented toward the stag.

This variation in social context enables assessment of how agent attack preferences change depending on the presence of other agents and the types of agents present. By comparing agent attack choices across these conditions, the study can reveal whether agents show different preferences when alone versus when paired with agents of different types, and how the type of partner agent influences attack target selection.

## Measurement

For each test condition, agents were placed at predetermined starting positions with default orientations. The focal agent started at the center position (row 7) facing west (left), while the fake partner agent (when present) started at the upper position (row 4) facing south (toward the stag). The test ran for a maximum number of turns (configurable via `max_test_steps`, typically 20-50 turns), during which the focal agent could move and attack freely while fake agents remained stationary.

The test measured the focal agent's first attack target during the test period. Attack tracking was performed using the environment's metrics collection system, which records when agents attack stags versus hares. The test identified the first successful attack made by the focal agent and recorded the target type. Results were encoded as:
- **1.0**: First attack targeted a stag
- **0.0**: First attack targeted a hare  
- **0.5**: No attack occurred within the test period

This measurement approach provides a direct behavioral measure of agent preference that reflects actual decision-making under conditions where agents can move, explore, and make multiple choices, rather than inferring preferences from initial action values alone.

## Data Collection

For each probe test, data was collected for every combination of agent, partner condition, and epoch. This resulted in a comprehensive dataset capturing agent attack preferences across different social contexts. The data included the first attack target type (stag, hare, or none), the numeric result (1.0, 0.0, or 0.5), and the turn number when the first attack occurred (or the maximum number of turns if no attack occurred).

Visualizations of the initial test state were saved for the first N probe tests (configurable via `save_png_for_first_n_tests`) to provide examples of the test conditions, while detailed numerical data was saved for all probe tests to enable comprehensive analysis of how agent attack preferences evolve over the course of training. The data was saved in CSV format with columns for epoch, agent_id, partner_kind, first_attack_target, result, and turn_of_first_attack.

## Summary

The multi-step probe test system used in this study measures agent attack preferences by observing actual behavioral choices over multiple turns in controlled test scenarios. Tests are conducted every 100 epochs throughout training using frozen copies of agents that do not learn during testing. Each agent is tested across three different social contexts (alone, with AgentKindA partner, with AgentKindB partner) using a single test map that contains both stag and hare resources. The test measures which resource type the agent attacks first, providing a direct behavioral measure of agent preference that reveals how agents balance the pursuit of high-value but challenging stags versus low-value but easy hares, and how these preferences may depend on social context. This approach enables tracking of how agent attack preferences develop over the course of training and how they vary across different social conditions, complementing the one-step test intention mode by providing behavioral validation of inferred preferences.


