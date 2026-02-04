from collections.abc import Sequence

from phylogenie.skyline import SkylineParameterLike, skyline_parameter
from phylogenie.tree_node import TreeNode
from phylogenie.treesimulator.events import StochasticEvent, TimedEvent
from phylogenie.treesimulator.model import Model

STATE_KEY = "state"


def _get_node_name(node_id: int, state: str) -> str:
    return f"{node_id}|{state}"


class UnboundedPopulationModel(Model):
    def __init__(
        self, init_state: str, max_time: float | None = None, seed: int | None = None
    ):
        super().__init__(max_time=max_time, seed=seed)
        self._init_state = init_state
        self.init()

    def init(self):
        super().init()
        self._next_node_id = 0
        self._sampled: set[str] = set()
        self._times: dict[TreeNode, float] = {}
        self._nodes: dict[int, TreeNode] = {}
        seed_individual = self._get_new_individual(self._init_state)
        self._tree = self._nodes[seed_individual]

    @property
    def tree_size(self) -> int:
        return len(self._sampled)

    def _get_new_node(self, state: str) -> TreeNode:
        self._next_node_id += 1
        node = TreeNode(_get_node_name(self._next_node_id, state))
        node[STATE_KEY] = state
        return node

    def _get_new_individual(self, state: str) -> int:
        new_id = self.create_new_individual(state)
        node = self._get_new_node(state)
        self._nodes[new_id] = node
        return new_id

    def _set_branch_length(self, node: TreeNode, time: float) -> None:
        if node.branch_length is not None:
            raise ValueError(f"Branch length of node {node.name} is already set.")
        parent_time = 0.0 if node.parent is None else self._times[node.parent]
        node.branch_length = time - parent_time
        self._times[node] = time

    def _stem(self, individual: int, time: float) -> None:
        node = self._nodes[individual]
        self._set_branch_length(node, time)
        stem_node = self._get_new_node(self._states[individual])
        node.add_child(stem_node)
        self._nodes[individual] = stem_node

    def remove(self, individual: int, time: float) -> None:
        self.remove_individual(individual)
        self._set_branch_length(self._nodes[individual], time)

    def migrate(self, individual: int, new_state: str, time: float) -> None:
        self.migrate_individual(individual, new_state)
        self._stem(individual, time)

    def birth_from(self, parent_individual: int, new_state: str, time: float) -> int:
        new_individual = self._get_new_individual(new_state)
        parent_node = self._nodes[parent_individual]
        new_node = self._nodes[new_individual]
        parent_node.add_child(new_node)
        self._stem(parent_individual, time)
        return new_individual

    def sample(self, individual: int, time: float, removal: bool) -> None:
        if removal:
            self._sampled.add(self._nodes[individual].name)
            self.remove(individual, time)
        else:
            sample_node = self._get_new_node(self._states[individual])
            sample_node.branch_length = 0.0
            self._sampled.add(sample_node.name)
            self._nodes[individual].add_child(sample_node)
            self._stem(individual, time)

    def get_tree(self) -> TreeNode | None:
        tree = self._tree.copy()
        for node in list(tree.iter_postorder()):
            if node.name not in self._sampled and not node.children:
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

    # --------------
    # Event handlers
    # --------------

    def add_birth_event(self, state: str, child_state: str, rate: SkylineParameterLike):
        def birth_event(time: float):
            self.birth_from(self.draw_individual(state), child_state, time)

        self._stochastic_events.append(
            StochasticEvent(
                fn=birth_event, reactants={state: 1}, rate=skyline_parameter(rate)
            )
        )

    def add_death_event(self, state: str, rate: SkylineParameterLike):
        def death_event(time: float):
            self.remove(self.draw_individual(state), time)

        self._stochastic_events.append(
            StochasticEvent(
                fn=death_event, reactants={state: 1}, rate=skyline_parameter(rate)
            )
        )

    def add_migration_event(
        self, state: str, target_state: str, rate: SkylineParameterLike
    ):
        def migration_event(time: float):
            self.migrate(self.draw_individual(state), target_state, time)

        self._stochastic_events.append(
            StochasticEvent(
                fn=migration_event,
                reactants={state: 1},
                rate=skyline_parameter(rate),
            )
        )

    def add_sampling_event(
        self,
        state: str,
        removal: bool,
        rate: SkylineParameterLike,
    ):
        def sampling_event(time: float):
            self.sample(self.draw_individual(state), time, removal)

        self._stochastic_events.append(
            StochasticEvent(
                fn=sampling_event, reactants={state: 1}, rate=skyline_parameter(rate)
            )
        )

    def add_timed_sampling_event(
        self,
        state: str,
        proportion: float,
        removal: bool,
        times: Sequence[float],
    ):
        def timed_sampling_event(time: float):
            for individual in self.get_individuals(state):
                if self._rng.random() < proportion:
                    self.sample(individual, time, removal)

        self._timed_events.append(
            TimedEvent(fn=timed_sampling_event, times=sorted(times))
        )
