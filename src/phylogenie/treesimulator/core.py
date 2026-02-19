import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Any

import numpy as np

import phylogenie._typings as pgt
from phylogenie._utils import MetadataMixin, OrderedSet
from phylogenie.skyline import SkylineParameter
from phylogenie.tree_node import TreeNode

STATE_KEY = "state"


@dataclass
class Event(ABC):
    @abstractmethod
    def apply(self, model: "Model"): ...

    def __call__(self, model: "Model"):
        self.apply(model)


@dataclass
class StochasticEvent(Event):
    rate: SkylineParameter

    @abstractmethod
    def get_propensity(self, model: "Model") -> float: ...


@dataclass
class TimedEvent(Event):
    times: Sequence[pgt.Scalar]


class Model(MetadataMixin):
    def __init__(
        self,
        init_state: str,
        init_metadata: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.rng = Random()
        self._init_state = init_state
        self._init_metadata = init_metadata
        self._stochastic_events: list[StochasticEvent] = []
        self._timed_events: list[TimedEvent] = []

    @staticmethod
    def _get_node_name(node_id: int, state: str) -> str:
        return f"{node_id}|{state}"

    def reset(self):
        self.clear()
        if self._init_metadata is not None:
            self._metadata.update(self._init_metadata)
        self._current_time = 0.0
        self._next_node_id = 0
        self._active_nodes: dict[str, OrderedSet[TreeNode]] = defaultdict(OrderedSet)
        self._sampled_nodes: set[str] = set()
        self._node_times: dict[TreeNode, float] = {}
        self._tree = self.get_new_node(self._init_state)
        self._run_stochastic_events = self._stochastic_events.copy()
        self._run_timed_events = self._timed_events.copy()

    @property
    def current_time(self) -> float:
        return self._current_time

    def get_new_node(self, state: str) -> TreeNode:
        self._next_node_id += 1
        node = TreeNode(self._get_node_name(self._next_node_id, state))
        node[STATE_KEY] = state
        self._active_nodes[state].add(node)
        return node

    def fix(self, node: TreeNode):
        if node.branch_length is not None:
            raise RuntimeError(f"Node {node} has already been fixed")
        parent_time = 0.0 if node.parent is None else self._node_times[node.parent]
        node.branch_length = self.current_time - parent_time
        self._node_times[node] = self.current_time
        self._active_nodes[node[STATE_KEY]].remove(node)

    def stem(self, node: TreeNode, stem_state: str):
        self.fix(node)
        stem_node = self.get_new_node(stem_state)
        node.add_child(stem_node)

    def remove(self, node: TreeNode):
        self.fix(node)

    def migrate(self, node: TreeNode, new_state: str):
        self.stem(node, new_state)

    def birth_from(self, parent: TreeNode, new_state: str) -> TreeNode:
        new_node = self.get_new_node(new_state)
        parent.add_child(new_node)
        self.stem(parent, parent[STATE_KEY])
        return new_node

    def sample(self, node: TreeNode, removal: bool) -> None:
        sample_node = node if removal else self.birth_from(node, node[STATE_KEY])
        self._sampled_nodes.add(sample_node.name)
        self.fix(sample_node)

    def get_tree(self) -> TreeNode | None:
        tree = self._tree.copy()
        for node in list(tree.iter_postorder()):
            if node.name not in self._sampled_nodes and not node.children:
                if node.parent is None:
                    return None
                else:
                    node.parent.remove_child(node)
            elif len(node.children) == 1:
                (child,) = node.children
                child.update_parent(node.parent)
                child.branch_length += node.branch_length  # pyright: ignore
                if node.parent is None:
                    return child
                else:
                    node.parent.remove_child(node)
        return tree

    @property
    def tree_size(self) -> int:
        return len(self._sampled_nodes)

    def get_active_nodes(self, state: str | None = None) -> list[TreeNode]:
        return [
            node
            for s, nodes in self._active_nodes.items()
            if state is None or re.fullmatch(state, s) is not None
            for node in nodes
        ]

    def count_active_nodes(self, state: str | None = None) -> int:
        return len(self.get_active_nodes(state))

    def draw_active_node(self, state: str | None = None) -> TreeNode:
        return self.rng.choice(self.get_active_nodes(state))

    def step(self, max_time: float | None = None) -> bool:
        # Get the next stochastic event change time
        rate_change_times = [
            t
            for e in self._run_stochastic_events
            for t in e.rate.change_times
            if t > self.current_time
        ]
        rate_change_time = min(rate_change_times) if rate_change_times else None

        # Get the next stochastic event time
        propensities = [e.get_propensity(self) for e in self._run_stochastic_events]
        total_propensity = sum(propensities)
        stochastic_event_time = (
            None
            if not total_propensity
            else self.current_time + self.rng.expovariate(total_propensity)
        )

        # Get the next timed events time
        timed_events_times = [
            t for e in self._run_timed_events for t in e.times if t > self._current_time
        ]
        timed_events_time = min(timed_events_times) if timed_events_times else None

        # If there are no more events to process, we stop the simulation.
        times = [rate_change_time, timed_events_time, stochastic_event_time]
        if all(t is None for t in times):
            return False

        self._current_time = min(t for t in [*times, max_time] if t is not None)

        if self._current_time == timed_events_time:
            for timed_event in [
                e for e in self._run_timed_events if timed_events_time in e.times
            ]:
                timed_event(self)

        if self._current_time == stochastic_event_time:
            self._run_stochastic_events[
                np.searchsorted(
                    np.cumsum(propensities) / total_propensity, self.rng.random()
                )
            ](self)

        if self._current_time == max_time:
            return False
        return True

    def add_stochastic_event(self, event: StochasticEvent):
        self._stochastic_events.append(event)

    def add_timed_event(self, event: TimedEvent):
        self._timed_events.append(event)

    def add_run_stochastic_event(self, event: StochasticEvent):
        self._run_stochastic_events.append(event)

    def add_run_timed_event(self, event: TimedEvent):
        self._run_timed_events.append(event)
