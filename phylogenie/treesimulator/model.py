from collections import defaultdict
from dataclasses import dataclass

from phylogenie.mixins import MetadataMixin
from phylogenie.treesimulator.tree import Tree


@dataclass
class Individual:
    id: int
    node: Tree
    state: str


def _get_node_name(node_id: int, state: str) -> str:
    return f"{node_id}|{state}"


def get_node_state(node_name: str) -> str:
    if "|" not in node_name:
        raise ValueError(
            f"Invalid node name: {node_name} (expected format 'id|state')."
        )
    return node_name.split("|")[-1]


class Model(MetadataMixin):
    def __init__(self, init_state: str):
        super().__init__()
        self._next_node_id = 0
        self._next_individual_id = 0
        self._population: dict[int, Individual] = {}
        self._states: dict[str, set[int]] = defaultdict(set)
        self._sampled: set[str] = set()
        self._tree = self._get_new_individual(init_state).node

    @property
    def n_sampled(self) -> int:
        return len(self._sampled)

    def _get_new_node(self, state: str) -> Tree:
        self._next_node_id += 1
        node = Tree(_get_node_name(self._next_node_id, state))
        return node

    def _get_new_individual(self, state: str) -> Individual:
        self._next_individual_id += 1
        individual = Individual(
            self._next_individual_id, self._get_new_node(state), state
        )
        self._population[individual.id] = individual
        self._states[state].add(individual.id)
        return individual

    def _set_branch_length(self, node: Tree, time: float) -> None:
        if node.branch_length is not None:
            raise ValueError(f"Branch length of node {node.name} is already set.")
        node.branch_length = time if node.parent is None else time - node.parent.depth

    def _stem(self, individual: Individual, time: float) -> None:
        self._set_branch_length(individual.node, time)
        stem_node = self._get_new_node(individual.state)
        individual.node.add_child(stem_node)
        individual.node = stem_node

    def remove(self, id: int, time: float) -> None:
        individual = self._population[id]
        self._set_branch_length(individual.node, time)
        self._population.pop(id)
        self._states[individual.state].remove(id)

    def migrate(self, id: int, state: str, time: float) -> None:
        individual = self._population[id]
        self._states[individual.state].remove(id)
        individual.state = state
        self._states[state].add(id)
        self._stem(individual, time)

    def birth_from(self, id: int, state: str, time: float) -> int:
        individual = self._population[id]
        new_individual = self._get_new_individual(state)
        individual.node.add_child(new_individual.node)
        self._stem(individual, time)
        return new_individual.id

    def sample(self, id: int, time: float, removal: bool) -> None:
        individual = self._population[id]
        if removal:
            self._sampled.add(individual.node.name)
            self.remove(id, time)
        else:
            sample_node = self._get_new_node(individual.state)
            sample_node.branch_length = 0.0
            self._sampled.add(sample_node.name)
            individual.node.add_child(sample_node)
            self._stem(individual, time)

    def get_state(self, id: int) -> str:
        return self._population[id].state

    def get_sampled_tree(self) -> Tree:
        tree = self._tree.copy()
        for node in list(tree.postorder_traversal()):
            if node.name not in self._sampled and not node.children:
                if node.parent is None:
                    raise ValueError("No samples in the tree.")
                else:
                    node.parent.remove_child(node)
            elif len(node.children) == 1:
                (child,) = node.children
                child.set_parent(node.parent)
                child.branch_length += node.branch_length  # pyright: ignore
                if node.parent is None:
                    return child
                else:
                    node.parent.remove_child(node)
        return tree

    def get_full_tree(self) -> Tree:
        return self._tree.copy()

    def get_population(self, states: str | None = None) -> list[int]:
        if states is None:
            return list(self._population)
        return list(self._states[states])

    def count_individuals(self, states: str | None = None) -> int:
        if states is None:
            return len(self._population)
        return len(self._states[states])
