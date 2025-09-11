import pprint

from sorrel.entities.entity import Entity


class Node:
    """Nodes are a class that can contain an indefinite number of entities and agents.
    They connect to other nodes through being _adjacent_ to these nodes; entities at
    other nodes can also be _visible_ in a particular node depending on a node's
    visibility.

    Attributes:
      name (str): The name of the node.
      agents (list[Agent]): A list of agents in the node.
      entities (list[Entity]): A list of entities in the node.
      adjacent (list[Node]): A list of adjacent nodes.
      visible (list[Node]): A list of visible nodes.
    """

    def __init__(self, name: str):
        """Initialize a node.

        Args:
          name (str): The name of the node.
        """
        self.name = name
        self.entities: list[Entity] = []
        self.adjacent: list[Node] = []
        self.visible: list[Node] = []

    def __str__(self):
        return pprint.pformat(
            {
                "name": self.name,
                "entities": self.entities,
                "visible_objects": self._print_visible_helper(),
                "adjacent_nodes": [node.name for node in self.adjacent],
            },
            compact=True,
        )

    def __repr__(self):
        return str(self)

    def _print_visible_helper(self) -> list[dict[str, Entity]]:
        """Return a list of visible entities.

        Returns:
          list: A list of dict entries mapping a node name and
          Entity at that location.
        """
        visible_obs = []
        for node in self.visible:
            for entity in node.entities:
                visible_obs.append({node.name, entity})
        return visible_obs

    def get_adjacent(self) -> list[str]:
        return [n.name for n in self.adjacent]

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to this node.

        Args:
          entity (Entity): An entity to add to this node.
        """
        self.entities.append(entity)

    def remove_entity(self, entity: Entity) -> Entity | None:
        """Remove an entity from this node.

        Args:
          entity (Entity): An entity to remove from this node.
        """
        for i, each_entity in enumerate(self.entities):
            if each_entity == entity:
                return self.entities.pop(i)

    def add_visible(self, node: "Node") -> None:
        """Add a visible node.

        Args:
          node (Node): A node that is visible to this node.
        """
        self.visible.append(node)

    def add_adjacent(self, node: "Node") -> None:
        """Add an adjacent node.

        Args:
          node (Node): A node that is adjacent to this node.
        """
        if node not in self.adjacent:
            self.adjacent.append(node)


class NodeWorld:
    """World that represents a graph of nodes. Each node can contain agents and
    entities.

    Attributes:
      struct (dict[str, Node]): The node structure of the world.
    """

    def __init__(
        self,
        num_nodes: int,
        adjacencies: list[list[str]],
        visibilities: dict[str, list[str]],
    ):
        """Initialize the node structure."""
        # Create node structure
        self.struct: dict[str, Node] = {}
        for i in range(1, num_nodes + 1):
            name = f"node_{i}"
            self.struct[name] = Node(name=name)
        for adjacency in adjacencies:
            self.set_adjacent(*adjacency)
        for node_name, visible_from in visibilities.items():
            self.set_visibility(node_name, visible_from)

    def __str__(self):
        return pprint.pformat(self.struct, compact=True)

    def __repr__(self):
        return str(self)

    def __getitem__(self, idx):
        return self.struct[idx]

    def add(self, target_location: str, entity: Entity):
        for node_name, node in self.struct.items():
            if node_name == target_location:
                entity.location = (target_location,)
                node.add_entity(entity)

    def remove(self, target_location: str) -> Entity | None:
        for node_name, node in self.struct.items():
            if node_name == target_location:
                for entity in node.entities:
                    entity = node.remove_entity(entity)
                    if entity is not None:
                        entity.location = (None,)
                    return entity

    def move(self, entity: Entity, new_location: str) -> bool:
        for node_name, node in self.struct.items():
            if (node_name,) == entity.location:
                tmp = node.remove_entity(entity)
                node.add_entity(entity)
                return True
        return False

    def set_visibility(self, location: str, visible_from: list[str]) -> None:
        for node, value in self.struct.items():
            if node in visible_from:
                value.add_visible(self.struct[location])

    def set_adjacent(self, node_a: str, node_b: str):
        # Set adjacency for nodes A and B.
        self.struct[node_a].add_adjacent(self.struct[node_b])
        self.struct[node_b].add_adjacent(self.struct[node_a])
