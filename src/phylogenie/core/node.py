from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

from phylogenie.mixins import MetadataMixin


class Node(MetadataMixin):
    def __init__(self, name: str = "", branch_length: float | None = None):
        super().__init__()
        self.name = name
        self.branch_length = branch_length
        self._parent: Node | None = None
        self._children: list[Node] = []

    # ----------------
    # Basic properties
    # ----------------
    # Properties related to parent-child relationships.

    @property
    def children(self) -> tuple["Node", ...]:
        return tuple(self._children)

    @property
    def parent(self) -> "Node | None":
        return self._parent

    def add_child(self, child: "Node") -> "Node":
        if child.parent is not None:
            raise ValueError(f"Node {child.name} already has a parent.")
        child._parent = self
        self._children.append(child)
        return self

    def remove_child(self, child: "Node") -> None:
        self._children.remove(child)
        child._parent = None

    def set_parent(self, parent: "Node | None"):
        if self.parent is not None:
            self.parent.remove_child(self)
        self._parent = parent
        if parent is not None:
            parent._children.append(self)

    def is_leaf(self) -> bool:
        return not self.children

    def is_internal(self) -> bool:
        return not self.is_leaf()

    def branch_length_or_raise(self) -> float:
        if self.parent is None:
            return 0 if self.branch_length is None else self.branch_length
        if self.branch_length is None:
            raise ValueError(f"Branch length of node {self.name} is not set.")
        return self.branch_length

    # --------------
    # Tree traversal
    # --------------
    # Methods for traversing the tree in various orders.

    def iter_ancestors(self, stop: "Node | None" = None) -> Iterator["Node"]:
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

    def iter_upward(self, stop: "Node | None" = None) -> Iterator["Node"]:
        if self == stop:
            return
        yield self
        yield from self.iter_ancestors(stop=stop)

    def iter_descendants(self) -> Iterator["Node"]:
        for child in self.children:
            yield child
            yield from child.iter_descendants()

    def iter_preorder(self) -> Iterator["Node"]:
        yield self
        yield from self.iter_descendants()

    def iter_inorder(self) -> Iterator["Node"]:
        if self.is_leaf():
            yield self
            return
        if len(self.children) != 2:
            raise ValueError("Inorder traversal is only defined for binary trees.")
        left, right = self.children
        yield from left.iter_inorder()
        yield self
        yield from right.iter_inorder()

    def iter_postorder(self) -> Iterator["Node"]:
        for child in self.children:
            yield from child.iter_postorder()
        yield self

    def iter_breadth_first(self) -> Iterator["Node"]:
        queue: deque["Node"] = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    # --------------
    # Paths and hops
    # --------------

    def get_mrca(self, node2: "Node") -> "Node":
        self_ancestors = set(self.iter_upward())
        for node2_ancestor in node2.iter_upward():
            if node2_ancestor in self_ancestors:
                return node2_ancestor
        raise ValueError(
            f"No common ancestor found between node {self} and node {node2}."
        )

    def get_path(self, node2: "Node") -> list["Node"]:
        mrca = self.get_mrca(node2)
        return [
            *self.iter_upward(stop=mrca.parent),
            *reversed(list(node2.iter_upward(stop=mrca))),
        ]

    def count_hops(self, node2: "Node") -> int:
        return len(self.get_path(node2)) - 1

    def get_distance(self, node2: "Node") -> float:
        mrca = self.get_mrca(node2)
        path = self.get_path(node2)
        path.remove(mrca)
        return sum(node.branch_length_or_raise() for node in path)

    # -------------
    # Miscellaneous
    # -------------
    # Miscellaneous methods like copying and representation.

    def copy(self):
        new_node = Node(self.name, self.branch_length)
        new_node.update(self.metadata)
        for child in self.children:
            new_node.add_child(child.copy())
        return new_node

    def sort_children(self, key: Callable[["Node"], Any]) -> None:
        self._children.sort(key=key)

    def __repr__(self) -> str:
        return f"Node(name='{self.name}', branch_length={self.branch_length}, metadata={self.metadata})"
