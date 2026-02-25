import math
from collections import deque
from collections.abc import Iterator, Mapping
from typing import Any

from phylogenie.utils import MetadataMixin


class TreeNode(MetadataMixin):
    """
    Phylogenetic tree node with parent-child relationships and metadata.

    This class represents both a node in a phylogenetic tree and the subtree
    rooted at that node. It supports parent-child relationships, tree traversal,
    various tree properties and metrics, and metadata annotations.
    """

    def __init__(self, name: str = "", branch_length: float | None = None):
        """Initialize a tree node with optional name and branch length."""
        super().__init__()
        self.name = name
        self.branch_length = branch_length
        self._parent: TreeNode | None = None
        self._children: list[TreeNode] = []

    # ----------------
    # Basic properties
    # ----------------
    # Properties related to parent-child relationships.

    @property
    def children(self) -> tuple["TreeNode", ...]:
        """Return the children of this node as a tuple."""
        return tuple(self._children)

    @property
    def parent(self) -> "TreeNode | None":
        """Return the parent of this node, if any."""
        return self._parent

    @property
    def root(self) -> "TreeNode":
        """Return the root node of the tree containing this node."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def add_child(self, child: "TreeNode") -> "TreeNode":
        """
        Attach a child node to this node.

        This updates both the child's parent pointer and this node's internal
        children list. A child may only have one parent.
        """
        if child.parent is not None:
            raise ValueError(f"Node {child.name} already has a parent.")
        child._parent = self
        self._children.append(child)
        return self

    def remove_child(self, child: "TreeNode"):
        """
        Detach a child node from this node.

        The child is removed from the children list and its parent pointer is
        cleared. The child must already be attached to this node.
        """
        if child not in self._children:
            raise ValueError(f"Node {child.name} is not a child of node {self.name}.")
        self._children.remove(child)
        child._parent = None

    def update_parent(self, parent: "TreeNode | None"):
        """
        Set or clear this node's parent, updating both sides of the link.

        If the node currently has a parent, it is removed from that parent's
        children list before the new relationship is created.
        """
        if self.parent is not None:
            self.parent.remove_child(self)
        self._parent = parent
        if parent is not None:
            parent._children.append(self)

    def is_leaf(self) -> bool:
        """Determine whether this node is a leaf (has no children)."""
        return not self.children

    def is_internal(self) -> bool:
        """Determine whether this node is internal (has one or more children)."""
        return not self.is_leaf()

    def get_leaves(self) -> tuple["TreeNode", ...]:
        """Return all leaf nodes in the subtree rooted at this node."""
        return tuple(node for node in self if node.is_leaf())

    def get_internal_nodes(self) -> tuple["TreeNode", ...]:
        """Return all internal (non-leaf) nodes in the subtree rooted at this node."""
        return tuple(node for node in self if node.is_internal())

    def is_binary(self) -> bool:
        """Determine whether the subtree rooted at this node is binary (every node has 0 or 2 children)."""
        return all(len(node.children) == 2 for node in self.get_internal_nodes())

    def branch_length_or_raise(self) -> float:
        """
        Return the branch length, raising if it is missing.

        The root node returns 0.0 when its branch length is None to provide a
        convenient origin length for distance calculations.
        """
        if self.parent is None:
            return 0 if self.branch_length is None else self.branch_length
        if self.branch_length is None:
            raise ValueError(f"Branch length of node {self.name} is not set.")
        return self.branch_length

    # --------------
    # Tree traversal
    # --------------
    # Methods for traversing the tree in various orders.

    def iter_ancestors(self, stop: "TreeNode | None" = None) -> Iterator["TreeNode"]:
        """Iterate over ancestors from the parent up to (but excluding) a stop node."""
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

    def iter_upward(self, stop: "TreeNode | None" = None) -> Iterator["TreeNode"]:
        """Iterate from this node upward through its ancestors, including self and up to (but excluding) a stop node."""
        if self == stop:
            return
        yield self
        yield from self.iter_ancestors(stop=stop)

    def iter_descendants(self) -> Iterator["TreeNode"]:
        """Iterate over all descendants of this node (excluding self)."""
        for child in self.children:
            yield child
            yield from child.iter_descendants()

    def iter_preorder(self) -> Iterator["TreeNode"]:
        """Iterate over nodes in preorder (self before descendants)."""
        yield self
        yield from self.iter_descendants()

    def iter_inorder(self) -> Iterator["TreeNode"]:
        """
        Iterate over nodes in inorder for binary trees.

        For non-binary trees, a ValueError is raised because inorder is not
        well-defined.
        """
        if self.is_leaf():
            yield self
            return

        if len(self.children) != 2:
            raise ValueError("Inorder traversal is only defined for binary trees.")

        left, right = self.children
        yield from left.iter_inorder()
        yield self
        yield from right.iter_inorder()

    def iter_postorder(self) -> Iterator["TreeNode"]:
        """Iterate over nodes in postorder (descendants before self)."""
        for child in self.children:
            yield from child.iter_postorder()
        yield self

    def iter_breadth_first(self) -> Iterator["TreeNode"]:
        """Iterate over nodes in breadth-first order."""
        queue: deque["TreeNode"] = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    # --------------
    # Paths and hops
    # --------------

    def get_mrca(self, other: "TreeNode") -> "TreeNode":
        """Find the most recent common ancestor (MRCA) of this node and another node."""
        self_ancestors = set(self.iter_upward())
        for other_ancestor in other.iter_upward():
            if other_ancestor in self_ancestors:
                return other_ancestor
        raise ValueError(
            f"No common ancestor found between node {self.name} and node {other.name}."
        )

    def get_path(self, other: "TreeNode") -> list["TreeNode"]:
        """
        Get the node-by-node path between this node and another node.

        The returned list starts at this node, travels up to the MRCA, and then
        down to the target node.
        """
        mrca = self.get_mrca(other)
        return [
            *self.iter_upward(stop=mrca.parent),
            *reversed(list(other.iter_upward(stop=mrca))),
        ]

    def count_hops(self, other: "TreeNode") -> int:
        """Count the number of edges between this node and another node."""
        return len(self.get_path(other)) - 1

    def get_distance(self, other: "TreeNode") -> float:
        """Compute the branch-length distance between this node and another node."""
        mrca = self.get_mrca(other)
        path = self.get_path(other)
        path.remove(mrca)
        return sum(node.branch_length_or_raise() for node in path)

    # ---------------
    # Tree properties
    # ---------------

    @property
    def leaf_counts(self) -> dict["TreeNode", int]:
        """Compute the number of leaves under each node for the subtree rooted at this node."""
        n_leaves: dict[TreeNode, int] = {}
        for node in self.iter_postorder():
            n_leaves[node] = sum(n_leaves[child] for child in node.children) or 1
        return n_leaves

    @property
    def n_leaves(self) -> int:
        """Return the number of leaf nodes for the subtree rooted at this node."""
        return len(self.get_leaves())

    @property
    def height_levels(self) -> dict["TreeNode", int]:
        """
        Compute the height level of all nodes for the subtree rooted at this node.

        The height level is defined as the number of edges from a node to its farthest leaf.
        It is 0 for leaf nodes, and increases by 1 for each level up the tree.
        """
        height_levels: dict[TreeNode, int] = {}
        for node in self.iter_postorder():
            height_levels[node] = (
                0
                if node.is_leaf()
                else max(1 + height_levels[child] for child in node.children)
            )
        return height_levels

    @property
    def height_level(self) -> int:
        """
        Return the height level of the node.

        The height level is defined as the number of edges from a node to its farthest leaf.
        It is 0 for leaf nodes, and increases by 1 for each level up the tree.
        """
        return self.height_levels[self]

    @property
    def heights(self) -> dict["TreeNode", float]:
        """
        Compute the height of all nodes for the subtree rooted at this node.

        The height is defined as the distance in branch-length units from a node to its farthest leaf.
        It is 0 for leaf nodes, and increases by the branch lengths up the tree.
        """
        heights: dict["TreeNode", float] = {}
        for node in self.iter_postorder():
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
    def height(self) -> float:
        """
        Return the height of the node.

        The height is defined as the distance in branch-length units from a node to its farthest leaf.
        It is 0 for leaf nodes, and increases by the branch lengths up the tree.
        """
        return self.heights[self]

    @property
    def depth_levels(self) -> dict["TreeNode", int]:
        """
        Compute the depth level of all nodes for the subtree rooted at this node.

        The depth level is defined as the number of edges from a node to the current node.
        It is 0 for the current node, and increases by 1 for each level down the tree.
        """
        depth_levels: dict[TreeNode, int] = {self: 0}
        for node in self.iter_descendants():
            depth_levels[node] = depth_levels[node.parent] + 1  # pyright: ignore
        return depth_levels

    @property
    def depth_level(self) -> int:
        """
        Return the depth level of the node.

        The depth level is defined as the number of edges from a node to the root.
        It is 0 for the root, and increases by 1 for each level down the tree.
        """
        return self.count_hops(self.root)

    @property
    def depths(self) -> dict["TreeNode", float]:
        """
        Compute the depth of all nodes for the subtree rooted at this node.

        The depth is defined as the distance in branch-length units from a node to the current node.
        It is 0 for the current node, and increases by the branch lengths down the tree.
        """
        depths: dict[TreeNode, float] = {self: 0.0}
        for node in self.iter_descendants():
            parent_depth = depths[node.parent]  # pyright: ignore
            depths[node] = node.branch_length_or_raise() + parent_depth
        return depths

    @property
    def depth(self) -> float:
        """
        Compute the depth of this node.

        The depth is defined as the distance in branch-length units from a node to the root.
        It is 0 for the root, and increases by the branch lengths down the tree.
        """
        return self.get_distance(self.root)

    @property
    def times(self) -> dict["TreeNode", float]:
        """
        Compute the time of all nodes for the subtree rooted at this node.

        Times are measured forwards from the current node.
        The time of the current node is defined as its branch length,
        and increases by the branch lengths down the tree.
        """
        origin = self.branch_length_or_raise()
        return {node: origin + depth for node, depth in self.depths.items()}

    @property
    def time(self) -> float:
        """
        Compute the time of this node.

        Times are measured forwards from the root of the tree.
        The time of the root is defined as its branch length,
        and increases by the branch lengths down the tree.

        Returns
        --------
        float
            The time of this node.
        """
        root = self.root
        origin = root.branch_length_or_raise()
        return self.get_distance(root) + origin

    @property
    def ages(self) -> dict["TreeNode", float]:
        """
        Compute the age of all nodes for the subtree rooted at this node.

        Ages are measured backwards in time from the most recent leaf of the subtree in branch-length units.
        The age of the most recent leaf is defined as 0, and increases by the branch lengths up the tree.
        """
        ages: dict[TreeNode, float] = {self: self.height}
        for node in self.iter_descendants():
            ages[node] = ages[node.parent] - node.branch_length_or_raise()  # pyright: ignore
        return ages

    @property
    def age(self) -> float:
        """
        Return the age of this node.

        Ages are measured backwards in time from the most recent leaf of the tree in branch-length units.
        The age of the most recent leaf is defined as 0, and increases by the branch lengths up the tree.
        """
        return self.root.height - self.get_distance(self.root)

    @property
    def origin(self) -> float:
        """
        Return the origin time of the subtree rooted at this node.

        The origin time is defined as the age of the current node plus its branch length.
        """
        return self.age + self.branch_length_or_raise()

    def compute_sackin_index(self, normalize: bool = False) -> float:
        """
        Compute the Sackin index for the subtree rooted at this node.

        Parameters
        -----------
        normalize : bool, optional
            If the normalize flag is set to True, the Sackin index is normalized to the range [0, 1] for binary trees,
            where 0 corresponds to a perfectly balanced tree and 1 corresponds to a completely unbalanced (pectinate) tree.
            For non-binary trees, normalization is not defined and a ValueError is raised. Default is False.
        """
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

    # -------------
    # Miscellaneous
    # -------------
    # Miscellaneous methods like copying and representation.

    def copy(self) -> "TreeNode":
        """
        Deep-copy this node and its descendants.

        The returned node is a new tree with duplicated metadata and structure.
        """
        new_node = TreeNode(self.name, self.branch_length)
        new_node.update(self.metadata)
        for child in self.children:
            new_node.add_child(child.copy())
        return new_node

    def ladderize(self, key: Mapping["TreeNode", Any] | None = None):
        """
        Sort children of all nodes to produce a ladderized tree.

        Parameters
        -----------
        key : Mapping["TreeNode", Any] | None, optional
            Mapping used to sort children. Defaults to the number of leaf descendants of each node.
        """
        if key is None:
            key = self.leaf_counts

        for node in self:
            node._children.sort(key=lambda child: key[child])

    def get_descendant(self, name: str) -> "TreeNode":
        """Find the first descendant node with the given name."""
        for node in self:
            if node.name == name:
                return node
        raise ValueError(f"Node {name} not found.")

    # -------------
    # Dunder methods
    # -------------

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        return f"TreeNode(name='{self.name}', branch_length={self.branch_length}, metadata={self.metadata})"

    def __iter__(self) -> Iterator["TreeNode"]:
        """Iterate over all nodes in the subtree rooted at this node in preorder."""
        return self.iter_preorder()

    def __len__(self) -> int:
        """Return the number of nodes in the subtree rooted at this node."""
        return sum(1 for _ in self)
