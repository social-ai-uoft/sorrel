# Probe Test Description

## Overview

This study employed probe tests to measure agent intentions and preferences at regular intervals throughout training. Probe tests create frozen copies of agents at specific training epochs and evaluate their decision-making in controlled test environments without allowing the agents to learn or update their parameters. This approach enables assessment of agent capabilities and preferences at different stages of training without interfering with the ongoing learning process.

## Test Intention Probe Test

The study used test intention mode, which measures agent preferences and intentions by examining the action values that agents assign to different actions in controlled scenarios. In this mode, agents are placed in carefully designed test environments with specific spatial arrangements containing both stags and hares positioned in ways that require agents to make directional choices. The test measures the values agents assign to actions that would orient them toward stags versus actions that would orient them toward hares, revealing their preferences for different resource types and their expectations about coordination opportunities.

## Test Procedure

Probe tests were executed every 100 epochs throughout training. When a probe test was triggered, frozen copies of all agents were created. These frozen copies were identical to the original agents in terms of their learned parameters and decision-making models, but were set to evaluation mode with exploration disabled, ensuring that the tests measured the agents' learned policies rather than their exploratory behavior.

Each agent was tested across four different test maps, each with a distinct spatial layout designed to assess agent intentions in different contexts. For each map, agents were tested at two different spawn positions (upper and lower positions), providing multiple perspectives on agent decision-making in the same spatial configuration. The test environments were smaller than the training environment (7×7 units) and contained precisely positioned stags and hares that created clear directional choices for agents.

## Social Context Variation

To assess how social context influences agent intentions, each agent was tested under three different partner conditions. In the first condition, agents were tested alone with no partner agent present. In the second condition, agents were tested with a partner agent of the same type (AgentKindA or AgentKindB, matching the focal agent's type). In the third condition, agents were tested with a partner agent of a different type (AgentKindA if the focal agent was AgentKindB, or AgentKindB if the focal agent was AgentKindA). Partner agents were configured with the ability to hunt resources, allowing them to participate in resource collection.

This variation in social context enables assessment of how agent intentions change depending on the presence of other agents and the types of agents present. By comparing agent preferences across these conditions, the study can reveal whether agents show different preferences when alone versus when paired with agents that can or cannot hunt stags, and how the type of partner agent influences decision-making.

## Measurement

For each test condition, agents were placed at predetermined starting positions with predetermined orientations specified in an orientation reference file. The test measured the Q-values that agents assigned to all available actions (forward, backward, step left, step right, and attack) based on their current observation of the test environment. From these Q-values, softmax weights were calculated to provide normalized measures of agent preferences that are independent of the absolute magnitude of the action values.

The test specifically identified which actions would orient the agent toward the stag versus toward the hare based on the agent's starting position and orientation. The softmax weights for actions facing toward the stag and actions facing toward the hare were recorded as measures of agent preference. These weights indicate the relative probability the agent would choose to pursue a stag versus a hare, providing a direct measure of agent intention that is independent of the specific action values.

## Data Collection

For each probe test, data was collected for every combination of agent, test map, partner condition, and spawn position. This resulted in a comprehensive dataset capturing agent intentions across multiple contexts and social conditions. The data included the Q-values for all actions and the derived weights for facing toward stags versus hares. Visualizations of the test environments were saved for the first three probe tests to provide examples of the test conditions, while detailed numerical data was saved for all probe tests to enable comprehensive analysis of how agent intentions evolve over the course of training.

## Summary

The probe test system used in this study measures agent intentions by examining the action values agents assign to different directional choices in controlled test scenarios. Tests are conducted every 100 epochs throughout training using frozen copies of agents that do not learn during testing. Each agent is tested across four different spatial layouts, three different social contexts (alone, with same-type partner, with different-type partner), and two different spawn positions per map. The test measures the relative values agents assign to actions that orient them toward stags versus hares, providing measures of agent preference that reveal how agents balance the pursuit of high-value but challenging stags versus low-value but easy hares, and how these preferences may depend on social context. This approach enables tracking of how agent intentions develop over the course of training and how they vary across different contexts and social conditions.

