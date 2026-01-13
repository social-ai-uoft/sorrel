# Agent Observation System

## Methods

### Observation Architecture

Agents in our multi-agent environment receive observations that combine local visual information, agent identity encoding, internal state features, and spatial position information. The observation system is designed to provide agents with both perceptual information about their immediate surroundings and contextual information about their own state and location within the environment.

### Visual Field Representation

The core of the observation system is a **visual field**—a local, egocentric view of the environment centered on the observing agent. The visual field is a square grid with side length `2r + 1`, where `r` is the agent's vision radius. Each cell in this grid can contain various types of entities, such as walls, empty spaces, resources (stags and hares), terrain features (sand), and other agents.

To encode what is present in each cell, we use a **one-hot encoding scheme** called entity codes. In this scheme, each type of entity in the environment is assigned a unique channel in a multi-channel representation. For example, if the environment contains 10 different entity types (e.g., Empty, Wall, StagResource, HareResource, Sand, Agent, etc.), then each cell in the visual field is represented by a 10-dimensional vector where exactly one position is set to 1.0 (indicating the presence of that entity type) and all other positions are 0.0. This creates a sparse, categorical representation where each cell's contents are unambiguously identified.

The visual field is structured as a three-dimensional tensor with dimensions `(C, H, W)`, where `C` is the number of entity type channels, `H` is the height of the visual field (equal to `2r + 1`), and `W` is the width (also `2r + 1`). When an agent looks at a particular location in the visual field, it can determine what entity type is present there by examining which channel has a value of 1.0 at that spatial position.

### Agent Identity Encoding

Beyond simply detecting that another agent is present in a cell, agents can also observe **identity information** about other agents within their visual field. This identity encoding system allows agents to distinguish between different individual agents, recognize which agents belong to the same group or kind, and determine the orientation (facing direction) of other agents.

The identity encoding is implemented as additional channels that are concatenated with the entity type channels in the visual field. For each cell in the visual field, if an agent entity is present, that cell's identity channels contain information about that specific agent. If no agent is present (e.g., the cell contains a resource, wall, or is empty), the identity channels are set to a special "not applicable" (N/A) code indicating that identity information is not relevant for that cell.

The identity code itself is a concatenated vector composed of three components:

1. **Agent ID component**: A one-hot vector of length `N + 1`, where `N` is the total number of agents in the environment. The first `N` positions correspond to individual agent identifiers (agent 0, agent 1, agent 2, etc.), and the final position is an N/A flag. When an agent is present, exactly one of the first `N` positions is set to 1.0, uniquely identifying that specific agent. When no agent is present, the N/A flag is set to 1.0.

2. **Agent kind/group component**: A one-hot vector encoding the agent's group membership or kind (e.g., "AgentKindA" or "AgentKindB"). This allows agents to recognize whether another agent belongs to their own group or a different group. The length of this vector is `K + 1`, where `K` is the number of unique agent kinds, plus one N/A position for non-agent entities.

3. **Orientation component**: A one-hot vector of length 5 encoding the agent's facing direction. The first four positions correspond to the four cardinal directions (North, East, South, West), and the fifth position is an N/A flag for non-agent entities.

For example, in an environment with 4 agents and 2 agent kinds, the identity code for agent 0 (who belongs to AgentKindA and is facing North) would be a 13-dimensional vector: `[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]`, where the first 5 values encode the agent ID (position 0 = 1.0 for agent 0), the next 3 values encode the agent kind (position 5 = 1.0 for AgentKindA), and the final 5 values encode orientation (position 8 = 1.0 for North).

This identity encoding is spatially aligned with the visual field: when an agent observes another agent at a particular location in their visual field, the identity channels at that same location contain the complete identity information for that observed agent. This design ensures that identity information is directly associated with spatial position, making it straightforward for agents to learn associations between agent identities and their locations.

### Internal State Features

In addition to the visual and identity information, agents also receive four scalar values that encode their own internal state:

1. **Stag inventory count**: The number of stag resources currently held by the agent.
2. **Hare inventory count**: The number of hare resources currently held by the agent.
3. **Ready flag**: A binary indicator (0 or 1) that is set to 1 when the agent has at least one resource in their inventory, indicating readiness to perform certain actions.
4. **Interaction reward flag**: A binary indicator that is set to 1 if the agent received a reward from an interaction in the previous time step, providing temporal feedback about recent positive outcomes.

These features are appended as a flat vector to the observation, allowing agents to condition their behavior on their own resource holdings and recent reward history.

### Positional Embedding

To provide agents with information about their absolute location within the environment, we include a **positional embedding** that encodes the agent's current coordinates. This embedding uses a sinusoidal encoding scheme that represents spatial coordinates at multiple frequency scales.

For each coordinate dimension (x and y), the embedding computes sine and cosine values at progressively increasing frequencies. Specifically, for a given coordinate value and a specified scale parameter (typically 3), the embedding generates 3 pairs of sine/cosine values, each at a different frequency. This multi-scale representation allows the encoding to capture both fine-grained positional information (high frequencies) and coarse-grained positional information (low frequencies).

The positional embedding has a total size of `2 × (scale_x + scale_y)` values. With a default scale of 3 for both dimensions, this results in 12 values: 6 values encoding the x-coordinate (3 sine/cosine pairs) and 6 values encoding the y-coordinate (3 sine/cosine pairs). This sinusoidal encoding has the advantage of being periodic and smooth, which can help neural networks learn spatial relationships more effectively than raw coordinate values.

### Complete Observation Structure

The final observation vector is constructed by concatenating all components in the following order:

1. **Flattened visual field**: The entity type channels and identity channels are first concatenated along the channel dimension, then the resulting tensor is flattened into a one-dimensional vector. The size of this vector is `(C_entity + C_identity) × H × W`, where `C_entity` is the number of entity type channels, `C_identity` is the number of identity channels per cell, and `H` and `W` are the height and width of the visual field.

2. **Internal state features**: The four scalar values (stag inventory, hare inventory, ready flag, interaction reward flag) are appended.

3. **Positional embedding**: The 12-dimensional positional encoding is appended.

The total observation size is therefore `(C_entity + C_identity) × (2r + 1)² + 4 + 12` values. For example, with 10 entity types, 13 identity channels, and a vision radius of 4, the visual field contributes `(10 + 13) × 9² = 1,863` values, plus 4 internal state features and 12 positional embedding values, resulting in a total observation size of 1,879 values.

This observation structure provides agents with comprehensive information about their local environment (what entities are present and where), social context (which specific agents are nearby and their group affiliations), internal state (resource holdings and readiness), and spatial context (absolute position in the world), enabling rich and context-aware decision-making in the multi-agent environment.

