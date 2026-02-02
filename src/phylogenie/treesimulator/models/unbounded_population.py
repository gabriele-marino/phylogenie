from phylogenie.tree_node import TreeNode
from phylogenie.treesimulator.models.base import Model


def _get_node_name(node_id: int, state: str) -> str:
    return f"{node_id}|{state}"


def get_node_state(node_name: str) -> str:
    if "|" not in node_name:
        raise ValueError(
            f"Invalid node name: {node_name} (expected format 'id|state')."
        )
    return node_name.split("|")[-1]


class UnboundedPopulationModel(Model):
    def __init__(self, init_state: str):
        super().__init__()
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
