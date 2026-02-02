from abc import ABC, abstractmethod
from collections import defaultdict

from phylogenie._mixins import MetadataMixin
from phylogenie.tree_node import TreeNode


class Model(MetadataMixin, ABC):
    def init(self):
        self.clear()
        self._next_individual_id = 0
        self._population: dict[str, set[int]] = defaultdict(set)
        self._states: dict[int, str] = {}

    def create_new_individual(self, state: str) -> int:
        self._next_individual_id += 1
        self._population[state].add(self._next_individual_id)
        self._states[self._next_individual_id] = state
        return self._next_individual_id

    def remove_individual(self, individual: int) -> None:
        state = self._states[individual]
        self._population[state].remove(individual)
        del self._states[individual]

    def migrate_individual(self, individual: int, new_state: str) -> None:
        old_state = self._states[individual]
        self._population[old_state].remove(individual)
        self._states[individual] = new_state
        self._population[new_state].add(individual)

    def get_individuals(self, state: str | None = None) -> list[int]:
        if state is None:
            return list(self._states)
        return list(self._population[state])

    def count_individuals(self, state: str | None = None) -> int:
        return len(self.get_individuals(state))

    @property
    @abstractmethod
    def tree_size(self) -> int: ...

    @abstractmethod
    def get_tree(self) -> TreeNode | None: ...
