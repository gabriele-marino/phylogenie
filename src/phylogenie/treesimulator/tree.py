from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

from phylogenie.mixins import MetadataMixin


class Tree(MetadataMixin):
    def __init__(self, name: str = "", branch_length: float | None = None):
        super().__init__()
        self.name = name
        self.branch_length = branch_length
        self._parent: Tree | None = None
        self._children: list[Tree] = []

    # ----------------
    # Basic properties
    # ----------------
    # Properties related to parent-child relationships.

    @property
    def children(self) -> tuple["Tree", ...]:
        return tuple(self._children)

    @property
    def parent(self) -> "Tree | None":
        return self._parent

    def add_child(self, child: "Tree") -> "Tree":
        if child.parent is not None:
            raise ValueError(f"Node {child.name} already has a parent.")
        child._parent = self
        self._children.append(child)
        return self

    def remove_child(self, child: "Tree") -> None:
        self._children.remove(child)
        child._parent = None

    def set_parent(self, parent: "Tree | None"):
        if self.parent is not None:
            self.parent.remove_child(self)
        self._parent = parent
        if parent is not None:
            parent._children.append(self)

    def is_leaf(self) -> bool:
        return not self.children

    def get_leaves(self) -> tuple["Tree", ...]:
        return tuple(node for node in self if node.is_leaf())

    def is_internal(self) -> bool:
        return not self.is_leaf()

    def get_internal_nodes(self) -> tuple["Tree", ...]:
        return tuple(node for node in self if node.is_internal())

    def is_binary(self) -> bool:
        return all(len(node.children) in (0, 2) for node in self)

    # --------------
    # Tree traversal
    # --------------
    # Methods for traversing the tree in various orders.

    def iter_ancestors(self, stop: "Tree | None" = None) -> Iterator["Tree"]:
        node = self
        while True:
            if node.parent is None:
                if stop is None:
                    return
                raise ValueError("Reached root without encountering stop node.")
            node = node.parent
            if node == stop:
                return
            yield node

    def iter_upward(self, stop: "Tree | None" = None) -> Iterator["Tree"]:
        if self == stop:
            return
        yield self
        yield from self.iter_ancestors(stop=stop)

    def iter_descendants(self) -> Iterator["Tree"]:
        for child in self.children:
            yield child
            yield from child.iter_descendants()

    def preorder_traversal(self) -> Iterator["Tree"]:
        yield self
        yield from self.iter_descendants()

    def inorder_traversal(self) -> Iterator["Tree"]:
        if self.is_leaf():
            yield self
            return
        if len(self.children) != 2:
            raise ValueError("Inorder traversal is only defined for binary trees.")
        left, right = self.children
        yield from left.inorder_traversal()
        yield self
        yield from right.inorder_traversal()

    def postorder_traversal(self) -> Iterator["Tree"]:
        for child in self.children:
            yield from child.postorder_traversal()
        yield self

    def breadth_first_traversal(self) -> Iterator["Tree"]:
        queue: deque["Tree"] = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    # ---------------
    # Tree properties
    # ---------------
    # Properties and methods related to tree metrics like leaf count, depth, height, etc.

    @property
    def n_leaves(self) -> int:
        return len(self.get_leaves())

    def branch_length_or_raise(self) -> float:
        if self.parent is None:
            return 0 if self.branch_length is None else self.branch_length
        if self.branch_length is None:
            raise ValueError(f"Branch length of node {self.name} is not set.")
        return self.branch_length

    @property
    def depth_level(self) -> int:
        return 0 if self.parent is None else self.parent.depth_level + 1

    @property
    def depth(self) -> float:
        parent_depth = 0 if self.parent is None else self.parent.depth
        return parent_depth + self.branch_length_or_raise()

    @property
    def height_level(self) -> int:
        if self.is_leaf():
            return 0
        return 1 + max(child.height_level for child in self.children)

    @property
    def height(self) -> float:
        if self.is_leaf():
            return 0.0
        return max(
            child.branch_length_or_raise() + child.height for child in self.children
        )

    # -------------
    # Miscellaneous
    # -------------
    # Other useful miscellaneous methods.

    def ladderize(self, key: Callable[["Tree"], Any] | None = None) -> None:
        if key is None:
            key = lambda node: node.n_leaves
        self._children.sort(key=key)
        for child in self.children:
            child.ladderize(key)

    def get_node(self, name: str) -> "Tree":
        for node in self:
            if node.name == name:
                return node
        raise ValueError(f"Node with name {name} not found.")

    def copy(self):
        new_tree = Tree(self.name, self.branch_length)
        new_tree.update(self.metadata)
        for child in self.children:
            new_tree.add_child(child.copy())
        return new_tree

    # ----------------
    # Dunder methods
    # ----------------
    # Special methods for standard behaviors like iteration, length, and representation.

    def __iter__(self) -> Iterator["Tree"]:
        return self.preorder_traversal()

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        return f"TreeNode(name='{self.name}', branch_length={self.branch_length}, metadata={self.metadata})"
