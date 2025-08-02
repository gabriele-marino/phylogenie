from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar

from numpy.random import Generator, default_rng

from phylogenie.tree import Tree

CT_POSTFIX = "-CT"


def get_CT_state(state: str) -> str:
    return f"{state}{CT_POSTFIX}"


def is_CT_state(state: str) -> bool:
    return state.endswith(CT_POSTFIX)


@dataclass
class Individual:
    node: Tree
    state: str
    id: int = field(init=False)
    _id_counter: ClassVar[int] = 0

    def __post_init__(self):
        Individual._id_counter += 1
        self.id = Individual._id_counter


class Model:
    def __init__(
        self,
        init_state: str,
        max_notified_contacts: int = 1,
        notification_probability: float = 0,
        rng: int | Generator | None = None,
    ):
        self._next_node_id = 0
        self._population: dict[int, Individual] = {}
        self._states: dict[str, set[int]] = defaultdict(set)
        self._contacts: dict[int, list[Individual]] = defaultdict(list)
        self._sampled: set[str] = set()
        self._tree = self._get_new_individual(init_state).node
        self._max_notified_contacts = max_notified_contacts
        self._notification_probability = notification_probability
        self._rng = rng if isinstance(rng, Generator) else default_rng(rng)

    @property
    def n_sampled(self) -> int:
        return len(self._sampled)

    def _get_new_node(self, state: str) -> Tree:
        self._next_node_id += 1
        return Tree(f"{self._next_node_id}|{state}")

    def _get_new_individual(self, state: str) -> Individual:
        individual = Individual(self._get_new_node(state), state)
        self._population[individual.id] = individual
        self._states[state].add(individual.id)
        return individual

    def _set_branch_length(self, node: Tree, time: float) -> None:
        if node.branch_length is not None:
            raise ValueError(f"Branch length of node {node.id} is already set.")
        node.branch_length = (
            time if node.parent is None else time - node.parent.get_time()
        )

    def _stem(self, individual: Individual, time: float) -> None:
        self._set_branch_length(individual.node, time)
        stem_node = self._get_new_node(individual.state)
        individual.node.add_child(stem_node)
        individual.node = stem_node

    def remove(self, id: int, time: float) -> None:
        individual = self._population[id]
        self._set_branch_length(individual.node, time)
        state = individual.state
        self._population.pop(id)
        self._states[state].remove(id)

    def migrate(self, id: int, state: str, time: float) -> None:
        individual = self._population[id]
        self._states[individual.state].remove(id)
        individual.state = state
        self._states[state].add(id)
        self._stem(individual, time)

    def birth_from(self, id: int, state: str, time: float) -> None:
        individual = self._population[id]
        new_individual = self._get_new_individual(state)
        individual.node.add_child(new_individual.node)
        self._stem(individual, time)
        self._contacts[id].append(new_individual)
        self._contacts[new_individual.id].append(individual)

    def sample(self, id: int, time: float, removal_probability: float) -> None:
        individual = self._population[id]
        if self._rng.random() < removal_probability:
            self._sampled.add(individual.node.id)
            self.remove(id, time)
        else:
            sample_node = self._get_new_node(individual.state)
            sample_node.branch_length = 0.0
            self._sampled.add(sample_node.id)
            individual.node.add_child(sample_node)
            self._stem(individual, time)

        for contact in self._contacts[id][-self._max_notified_contacts :]:
            if (
                contact.id in self._population
                and not is_CT_state(contact.state)
                and self._rng.random() < self._notification_probability
            ):
                self.migrate(contact.id, get_CT_state(contact.state), time)

    def get_sampled_tree(self) -> Tree:
        tree = self._tree.copy()
        for node in list(tree.postorder_traversal()):
            if node.id not in self._sampled and not node.children:
                if node.parent is None:
                    raise ValueError("No samples in the tree.")
                else:
                    node.parent.children.remove(node)
            elif len(node.children) == 1:
                (child,) = node.children
                child.parent = node.parent
                assert child.branch_length is not None
                assert node.branch_length is not None
                child.branch_length += node.branch_length
                if node.parent is None:
                    return child
                else:
                    node.parent.children.append(child)
                    node.parent.children.remove(node)
        return tree

    def get_full_tree(self) -> Tree:
        return self._tree.copy()

    def get_random_individual(self, state: str | None = None) -> int:
        if state is None:
            return self._rng.choice(list(self._population))
        return self._rng.choice(list(self._states[state]))

    def get_population(self) -> list[int]:
        return list(self._population)

    def count_individuals(self, state: str | None = None) -> int:
        if state is None:
            return len(self._population)
        return len(self._states[state])
