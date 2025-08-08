from collections.abc import Iterator


class Tree:
    def __init__(self, id: str = "", branch_length: float | None = None):
        self.id = id
        self.branch_length = branch_length
        self.parent: Tree | None = None
        self.children: list[Tree] = []

    def add_child(self, child: "Tree") -> "Tree":
        child.parent = self
        self.children.append(child)
        return self

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

    def get_leaves(self) -> list["Tree"]:
        return [node for node in self if not node.children]

    def get_time(self) -> float:
        parent_time = 0 if self.parent is None else self.parent.get_time()
        if self.branch_length is None:
            raise ValueError(f"Branch length of node {self.id} is not set.")
        return self.branch_length + parent_time

    def is_leaf(self) -> bool:
        return not self.children

    def copy(self) -> "Tree":
        new_tree = Tree(self.id, self.branch_length)
        for child in self.children:
            new_tree.add_child(child.copy())
        return new_tree

    def __iter__(self) -> Iterator["Tree"]:
        return self.preorder_traversal()

    def __repr__(self) -> str:
        return f"TreeNode(id='{self.id}', branch_length={self.branch_length})"
