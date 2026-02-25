import re
from collections import defaultdict
from random import Random
from typing import Any, Protocol

from phylogenie.tree_node import TreeNode
from phylogenie.utils import MetadataMixin, OrderedSet

STATE = "state"


class Event(Protocol):
    def get_next_firing_time(self, model: "Model") -> float | None: ...
    def apply(self, model: "Model") -> None: ...


class Model(MetadataMixin):
    def __init__(self, init_state: str, init_metadata: dict[str, Any] | None = None):
        super().__init__()
        self.rng = Random()
        self._init_state = init_state
        self._init_metadata = init_metadata
        self._events: list[Event] = []

    @staticmethod
    def _get_node_name(node_id: int, state: str) -> str:
        return f"{node_id}|{state}"

    def reset(self):
        self.clear()
        if self._init_metadata is not None:
            self._metadata = self._init_metadata.copy()
        self._current_time = 0.0
        self._next_node_id = 0
        self._active_nodes: dict[str, OrderedSet[TreeNode]] = defaultdict(OrderedSet)
        self._sampled_nodes: set[str] = set()
        self._node_times: dict[TreeNode, float] = {}
        self._tree = self._get_new_node(self._init_state)
        self._run_events = self._events.copy()

    @property
    def current_time(self) -> float:
        return self._current_time

    def _get_new_node(self, state: str) -> TreeNode:
        self._next_node_id += 1
        node = TreeNode(self._get_node_name(self._next_node_id, state))
        node[STATE] = state
        self._active_nodes[state].add(node)
        return node

    def _fix(self, node: TreeNode):
        if node.branch_length is not None:
            raise RuntimeError(f"Node {node} has already been fixed")
        parent_time = 0.0 if node.parent is None else self._node_times[node.parent]
        node.branch_length = self.current_time - parent_time
        self._node_times[node] = self.current_time
        self._active_nodes[node[STATE]].remove(node)

    def _stem(self, node: TreeNode, stem_state: str) -> TreeNode:
        self._fix(node)
        stem_node = self._get_new_node(stem_state)
        node.add_child(stem_node)
        return stem_node

    def remove(self, node: TreeNode):
        self._fix(node)

    def migrate(self, node: TreeNode, new_state: str) -> TreeNode:
        return self._stem(node, new_state)

    def birth_from(self, parent: TreeNode, new_state: str) -> tuple[TreeNode, TreeNode]:
        new_node = self._get_new_node(new_state)
        parent.add_child(new_node)
        stem_node = self._stem(parent, parent[STATE])
        return stem_node, new_node

    def sample(self, node: TreeNode):
        self._sampled_nodes.add(node.name)
        self._fix(node)

    def get_sampled_tree(self) -> TreeNode | None:
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
    def n_sampled(self) -> int:
        return len(self._sampled_nodes)

    def get_active_nodes(self, state: str | None = None) -> tuple[TreeNode, ...]:
        return tuple(
            node
            for s, nodes in self._active_nodes.items()
            if state is None or re.fullmatch(state, s) is not None
            for node in nodes
        )

    def count_active_nodes(self, state: str | None = None) -> int:
        return len(self.get_active_nodes(state))

    def draw_active_node(self, state: str | None = None) -> TreeNode:
        return self.rng.choice(self.get_active_nodes(state))

    def step(self, max_time: float | None = None) -> bool:
        next_event_times = [e.get_next_firing_time(self) for e in self._run_events]
        next_step_times = [t for t in [*next_event_times, max_time] if t is not None]
        if not next_step_times:
            return False

        self._current_time = min(next_step_times)
        for event, time in zip(self._run_events, next_event_times):
            if time == self._current_time:
                event.apply(self)

        return self._current_time != max_time

    def add_event(self, event: Event):
        self._events.append(event)

    def add_run_event(self, event: Event):
        self._run_events.append(event)
