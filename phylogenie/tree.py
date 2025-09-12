from collections.abc import Iterator
from typing import Any


class Tree:
    def __init__(self, name: str = "", branch_length: float | None = None):
        self.name = name
        self.branch_length = branch_length
        self._parent: Tree | None = None
        self._children: list[Tree] = []
        self._features: dict[str, Any] = {}

    @property
    def children(self) -> tuple["Tree", ...]:
        return tuple(self._children)

    @property
    def parent(self) -> "Tree | None":
        return self._parent

    @property
    def features(self) -> dict[str, Any]:
        return self._features.copy()

    def add_child(self, child: "Tree") -> "Tree":
        child._parent = self
        self._children.append(child)
        return self

    def remove_child(self, child: "Tree") -> None:
        self._children.remove(child)
        child._parent = None

    def set_parent(self, node: "Tree | None"):
        self._parent = node
        if node is not None:
            node._children.append(self)

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

    def preorder_traversal(self) -> Iterator["Tree"]:
        yield self
        for child in self.children:
            yield from child.preorder_traversal()

    def postorder_traversal(self) -> Iterator["Tree"]:
        for child in self.children:
            yield from child.postorder_traversal()
        yield self

    def get_node(self, name: str) -> "Tree":
        for node in self:
            if node.name == name:
                return node
        raise ValueError(f"Node with name {name} not found.")

    def is_leaf(self) -> bool:
        return not self.children

    def get_leaves(self) -> tuple["Tree", ...]:
        return tuple(node for node in self if not node.children)

    def parse_branch_length(self) -> float:
        if self.branch_length is None:
            raise ValueError(f"Branch length of node {self.name} is not set.")
        return self.branch_length

    def get_time(self) -> float:
        parent_time = 0 if self.parent is None else self.parent.get_time()
        return self.parse_branch_length() + parent_time

    def set(self, key: str, value: Any) -> None:
        self._features[key] = value

    def update_features(self, features: dict[str, Any]) -> None:
        self._features.update(features)

    def get(self, key: str) -> Any:
        return self._features[key]

    def delete(self, key: str) -> None:
        del self._features[key]

    def copy(self):
        new_tree = Tree(self.name, self.branch_length)
        new_tree.update_features(self._features)
        for child in self.children:
            new_tree.add_child(child.copy())
        return new_tree

    def __iter__(self) -> Iterator["Tree"]:
        return self.preorder_traversal()

    def __repr__(self) -> str:
        return f"TreeNode(name='{self.name}', branch_length={self.branch_length}, features={self.features})"
