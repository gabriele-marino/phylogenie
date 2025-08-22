from collections.abc import Iterator
from typing import Any


class Tree:
    def __init__(self, id: str = "", branch_length: float | None = None):
        self.id = id
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
        return self._features

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

    def preorder_traversal(self) -> Iterator["Tree"]:
        yield self
        for child in self.children:
            yield from child.preorder_traversal()

    def postorder_traversal(self) -> Iterator["Tree"]:
        for child in self.children:
            yield from child.postorder_traversal()
        yield self

    def get_node(self, id: str) -> "Tree":
        for node in self:
            if node.id == id:
                return node
        raise ValueError(f"Node with id {id} not found.")

    def is_leaf(self) -> bool:
        return not self.children

    def get_leaves(self) -> list["Tree"]:
        return [node for node in self if not node.children]

    def get_time(self) -> float:
        parent_time = 0 if self.parent is None else self.parent.get_time()
        if self.branch_length is None:
            if self.parent is not None:
                raise ValueError(
                    f"Branch length of non-root node {self.id} is not set."
                )
            return 0.0
        return self.branch_length + parent_time

    def set(self, key: str, value: Any) -> None:
        self._features[key] = value

    def get(self, key: str) -> Any:
        return self._features.get(key)

    def delete(self, key: str) -> None:
        del self._features[key]

    def copy(self):
        new_tree = Tree(self.id, self.branch_length)
        for key, value in self._features.items():
            new_tree.set(key, value)
        for child in self.children:
            new_tree.add_child(child.copy())
        return new_tree

    def __iter__(self) -> Iterator["Tree"]:
        return self.preorder_traversal()

    def __repr__(self) -> str:
        return f"TreeNode(id='{self.id}', branch_length={self.branch_length}, features={self.features})"
