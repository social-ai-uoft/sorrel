# Probe Test Methods

## Overview

To assess agent intentions and behavioral preferences at different stages of training without interfering with the learning process, we employed probe tests that create frozen copies of agents at specific training epochs and evaluate their decision-making in controlled test environments. Probe tests were executed every 100 epochs throughout training. When a probe test was triggered, frozen copies of all agents were created. These frozen copies were identical to the original agents in terms of their learned parameters and decision-making models, but were set to evaluation mode with exploration disabled (epsilon = 0), ensuring that the tests measured the agents' learned policies rather than their exploratory behavior.

We implemented two complementary probe test modes: a one-step test intention mode that measures agent preferences through Q-value analysis, and a multi-step behavioral mode that measures agent preferences through observed attack choices. Both modes were designed to assess how social context influences agent decision-making by varying the presence and type of partner agents.

## Test Intention Probe Test (One-Step)

The test intention mode measures agent preferences and intentions by examining the action values that agents assign to different actions in controlled scenarios. In this mode, agents are placed in carefully designed test environments with specific spatial arrangements containing both stags and hares positioned in ways that require agents to make directional choices. The test measures the values agents assign to actions that would orient them toward stags versus actions that would orient them toward hares, revealing their preferences for different resource types and their expectations about coordination opportunities.

### Test Procedure

Each agent was tested across four different test maps, each with a distinct spatial layout designed to assess agent intentions in different contexts. For each map, agents were tested at two different spawn positions (upper and lower positions), providing multiple perspectives on agent decision-making in the same spatial configuration. The test environments were smaller than the training environment (7×7 units) and contained precisely positioned stags and hares that created clear directional choices for agents.

### Social Context Variation

To assess how social context influences agent intentions, each agent was tested under three different partner conditions. In the first condition, agents were tested alone with no partner agent present. In the second condition, agents were tested with a partner agent of the same type (AgentKindA or AgentKindB, matching the focal agent's type). In the third condition, agents were tested with a partner agent of a different type (AgentKindA if the focal agent was AgentKindB, or AgentKindB if the focal agent was AgentKindA). Partner agents were configured with the ability to hunt resources, allowing them to participate in resource collection.

### Measurement

For each test condition, agents were placed at predetermined starting positions with predetermined orientations specified in an orientation reference file. The test measured the Q-values that agents assigned to all available actions (forward, backward, step left, step right, and attack) based on their current observation of the test environment. From these Q-values, softmax weights were calculated to provide normalized measures of agent preferences that are independent of the absolute magnitude of the action values.

The test specifically identified which actions would orient the agent toward the stag versus toward the hare based on the agent's starting position and orientation. The softmax weights for actions facing toward the stag and actions facing toward the hare were recorded as measures of agent preference. These weights indicate the relative probability the agent would choose to pursue a stag versus a hare, providing a direct measure of agent intention that is independent of the specific action values.

## Multi-Step Probe Test (Behavioral)

The multi-step probe test mode measures agent attack preferences by observing their actual behavioral choices over multiple turns in controlled scenarios. Unlike the one-step test intention mode that measures Q-values at initialization, the multi-step test allows agents to move, explore, and make actual attack decisions, revealing their behavioral preferences through observed actions rather than inferred intentions. The test records the focal agent's first attack target (stag, hare, or none) during a configurable number of turns, encoding results as 1.0 for stag attacks, 0.0 for hare attacks, and 0.5 if no attack occurs within the test period.

### Test Procedure

Each agent was tested using the `test_multi_step.txt` map, which contains a focal agent spawn point at the center (row 7) and a fake agent spawn point at the upper position (row 4). The map also contains a stag resource at row 5 (above the focal agent) and a hare resource at row 9 (below the focal agent), creating a clear choice between pursuing the high-value stag or the low-value hare. The test environment is 13×13 units and contains precisely positioned resources that require agents to make directional and target selection choices.

The test ran for a maximum number of turns (configurable via `max_test_steps`, typically 15-50 turns), during which the focal agent could move and attack freely. Fake partner agents, when present, remained stationary and did not act during the test, serving as social context cues without interfering with the focal agent's decision-making process.

### Social Context Variation

To assess how social context influences agent attack preferences, each agent was tested under three different partner conditions, matching the test intention mode. In the first condition, agents were tested alone with no partner agent present. In the second condition, agents were tested with a fake partner agent of type AgentKindA positioned at the upper spawn point. In the third condition, agents were tested with a fake partner agent of type AgentKindB positioned at the upper spawn point.

The fake partner agent at the upper position (row 4) was oriented to face south (toward larger row numbers), directly facing the stag resource at row 5. This orientation ensures that the fake agent provides a clear social signal about the presence of the stag, allowing assessment of how the focal agent's attack preferences change depending on the type of partner agent present and whether the partner appears to be oriented toward the stag.

### Measurement

For each test condition, agents were placed at predetermined starting positions with default orientations. The focal agent started at the center position (row 7) facing west (left), while the fake partner agent (when present) started at the upper position (row 4) facing south (toward the stag).

The test measured the focal agent's first attack target during the test period. Attack tracking was performed using the environment's metrics collection system, which records when agents attack stags versus hares. The test identified the first successful attack made by the focal agent and recorded the target type. Results were encoded as:
- **1.0**: First attack targeted a stag
- **0.0**: First attack targeted a hare  
- **0.5**: No attack occurred within the test period

This measurement approach provides a direct behavioral measure of agent preference that reflects actual decision-making under conditions where agents can move, explore, and make multiple choices, rather than inferring preferences from initial action values alone.

## Data Collection

For both probe test modes, data was collected for every combination of agent, partner condition, and epoch. For the test intention mode, data was additionally collected for each test map and spawn position. This resulted in comprehensive datasets capturing agent intentions and behavioral preferences across multiple contexts and social conditions.

### Test Intention Data

The test intention mode data included the Q-values for all actions and the derived weights for facing toward stags versus hares. Data was saved in CSV format with columns for epoch, agent_id, map_name, partner_kind, version (spawn position), Q-values for each action, and weights for facing stag versus hare.

### Multi-Step Data

The multi-step mode data included the first attack target type (stag, hare, or none), the numeric result (1.0, 0.0, or 0.5), and the turn number when the first attack occurred (or the maximum number of turns if no attack occurred). Data was saved in CSV format with columns for epoch, agent_id, partner_kind, first_attack_target, result, and turn_of_first_attack.

## Complementary Analysis

The two probe test modes provide complementary insights into agent decision-making. The test intention mode measures inferred preferences through Q-value analysis at a single time point, revealing what agents value and intend to pursue. The multi-step mode measures observed behavioral choices over multiple turns, revealing what agents actually do when given the opportunity to move and act. By comparing results across both modes, we can assess the alignment between agent intentions (as measured by Q-values) and agent behavior (as measured by attack choices), providing a more complete picture of how agent preferences develop and how they are expressed in action.

