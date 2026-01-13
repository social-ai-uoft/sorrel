# Agent Observation (Brief Version for Journal Methods Section)

## Agent Observation

Agents receive observations that combine local visual information, agent identity encoding, internal state features, and spatial position information. The observation vector is constructed by concatenating four components: a flattened visual field, agent identity channels, internal state features, and a positional embedding.

**Visual Field.** Agents observe a local, egocentric view of the environment—a square grid with side length `2r + 1`, where `r` is the vision radius. Each cell in this grid is encoded using a one-hot scheme (entity codes), where each entity type (e.g., Empty, Wall, StagResource, HareResource, Sand, Agent) is assigned a unique channel. The visual field is structured as a three-dimensional tensor `(C, H, W)`, where `C` is the number of entity type channels, and `H = W = 2r + 1`.

**Agent Identity Encoding.** When agents are present in the visual field, additional identity channels encode three components: (1) a unique agent ID (one-hot vector of length `N + 1`, where `N` is the total number of agents), (2) agent group/kind membership (one-hot vector encoding agent kinds such as "AgentKindA" or "AgentKindB"), and (3) orientation (one-hot vector encoding the four cardinal directions). Identity channels are spatially aligned with the visual field—when an agent observes another agent at a particular location, the identity channels at that location contain the complete identity information for that observed agent. For non-agent entities, identity channels are set to a "not applicable" code.

**Internal State Features.** Four scalar values encode the agent's internal state: stag inventory count, hare inventory count, a binary ready flag (indicating whether the agent has at least one resource), and a binary interaction reward flag (indicating whether the agent received a reward in the previous time step).

**Positional Embedding.** The agent's absolute location within the environment is encoded using a sinusoidal positional embedding. For each coordinate dimension (x and y), the embedding computes sine and cosine values at multiple frequency scales (default scale: 3), resulting in 12 values total (6 per dimension). This multi-scale representation captures both fine-grained and coarse-grained positional information.

The final observation vector has size `(C_entity + C_identity) × (2r + 1)² + 4 + 12`, where `C_entity` is the number of entity type channels, `C_identity` is the number of identity channels per cell, and `r` is the vision radius. For example, with 10 entity types, 13 identity channels, and a vision radius of 4, the total observation size is 1,879 values.

