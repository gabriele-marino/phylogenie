import math
from collections.abc import Iterator, Mapping
from typing import Any

from phylogenie.core.node import Node
from phylogenie.mixins import MetadataMixin


class Tree(MetadataMixin):
    def __init__(self, root: Node):
        super().__init__()
        self.root = root

    # ----------------
    # Basic properties
    # ----------------
    # Properties related to parent-child relationships.

    def add_subtree(self, parent: Node, subtree: "Tree") -> "Tree":
        parent.add_child(subtree.root)
        return self

    def get_leaves(self) -> tuple[Node, ...]:
        return tuple(node for node in self if node.is_leaf())

    def get_internal_nodes(self) -> tuple[Node, ...]:
        return tuple(node for node in self if node.is_internal())

    def is_binary(self) -> bool:
        return all(len(node.children) in (0, 2) for node in self)

    # --------------
    # Tree traversal
    # --------------
    # Methods for traversing the tree in various orders.

    def preorder_traversal(self) -> Iterator[Node]:
        yield from self.root.iter_preorder()

    def inorder_traversal(self) -> Iterator[Node]:
        yield from self.root.iter_inorder()

    def postorder_traversal(self) -> Iterator[Node]:
        yield from self.root.iter_postorder()

    def breadth_first_traversal(self) -> Iterator[Node]:
        yield from self.root.iter_breadth_first()

    # ---------------
    # Tree properties
    # ---------------
    # Properties and methods related to tree metrics like leaf count, depth, height, etc.

    @property
    def n_leaves(self) -> int:
        return len(self.get_leaves())

    @property
    def leaf_counts(self) -> dict[Node, int]:
        n_leaves: dict[Node, int] = {}
        for node in self.postorder_traversal():
            n_leaves[node] = sum(n_leaves[child] for child in node.children) or 1
        return n_leaves

    @property
    def depth_levels(self) -> dict[Node, int]:
        depth_levels: dict[Node, int] = {self.root: 0}
        for node in self.root.iter_descendants():
            depth_levels[node] = depth_levels[node.parent] + 1  # pyright: ignore
        return depth_levels

    @property
    def depths(self) -> dict[Node, float]:
        depths: dict[Node, float] = {self.root: 0.0}
        for node in self.root.iter_descendants():
            parent_depth = depths[node.parent]  # pyright: ignore
            depths[node] = node.branch_length_or_raise() + parent_depth
        return depths

    @property
    def height_level(self) -> int:
        return self.height_levels[self.root]

    @property
    def times(self) -> dict[Node, float]:
        return self.depths

    @property
    def height_levels(self) -> dict[Node, int]:
        height_levels: dict[Node, int] = {}
        for node in self.postorder_traversal():
            height_levels[node] = (
                0
                if node.is_leaf()
                else max(1 + height_levels[child] for child in node.children)
            )
        return height_levels

    @property
    def height(self) -> float:
        return self.heights[self.root]

    @property
    def heights(self) -> dict[Node, float]:
        heights: dict[Node, float] = {}
        for node in self.postorder_traversal():
            heights[node] = (
                0
                if node.is_leaf()
                else max(
                    child.branch_length_or_raise() + heights[child]
                    for child in node.children
                )
            )
        return heights

    @property
    def age(self) -> float:
        return self.ages[self.root]

    @property
    def ages(self) -> dict[Node, float]:
        ages: dict[Node, float] = {self.root: self.height}
        for node in self.root.iter_descendants():
            ages[node] = ages[node.parent] - node.branch_length_or_raise()  # pyright: ignore
        return ages

    def compute_sackin_index(self, normalize: bool = False) -> float:
        sackin_index = sum(
            dl for node, dl in self.depth_levels.items() if node.is_leaf()
        )
        if normalize:
            if not self.is_binary():
                raise ValueError(
                    "Normalized Sackin index is only defined for binary trees."
                )
            n = self.n_leaves
            h = math.floor(math.log2(n))
            min_sackin_index = n * (h + 2) - 2 ** (h + 1)
            max_sackin_index = n * (n - 1) / 2
            return (sackin_index - min_sackin_index) / (
                max_sackin_index - min_sackin_index
            )
        return sackin_index

    def compute_mean_leaf_pairwise_distance(self) -> float:
        leaves = self.get_leaves()
        n_leaves = len(leaves)
        if n_leaves < 2:
            return 0.0

        total_distance = sum(
            leaves[i].get_distance(leaves[j])
            for i in range(n_leaves)
            for j in range(i + 1, n_leaves)
        )
        return total_distance / math.comb(n_leaves, 2)

    # -------------
    # Miscellaneous
    # -------------
    # Other useful miscellaneous methods.

    def ladderize(self, key: Mapping[Node, Any] | None = None) -> None:
        if key is None:
            key = self.leaf_counts

        for node in self:
            node.sort_children(key=lambda child: key[child])

    def get_node(self, name: str) -> Node:
        for node in self:
            if node.name == name:
                return node
        raise ValueError(f"Node {name} not found.")

    def copy(self):
        new_tree = Tree(self.root.copy())
        new_tree.update(self.metadata)
        return new_tree

    # ----------------
    # Dunder methods
    # ----------------
    # Special methods for standard behaviors like iteration, length, and representation.

    def __iter__(self) -> Iterator[Node]:
        return self.preorder_traversal()

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        return f"TreeNode(metadata={self.metadata})"
