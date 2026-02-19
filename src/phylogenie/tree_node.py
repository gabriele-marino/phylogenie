import math
from collections import deque
from collections.abc import Iterator, Mapping
from typing import Any

from phylogenie._utils import MetadataMixin


class TreeNode(MetadataMixin):
    """
    Phylogenetic tree node with parent-child relationships and metadata.

    This class represents both a node in a phylogenetic tree and the subtree
    rooted at that node. It supports parent-child relationships, tree traversal,
    various tree properties and metrics, and metadata annotations.
    """

    def __init__(self, name: str = "", branch_length: float | None = None):
        """
        Initialize a tree node with optional name and branch length.

        Parameters
        -----------
        name : str, optional
            The name of the node. Default is an empty string.
        branch_length : float | None, optional
            The length of the branch leading to this node. Default is None.
        """
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
        """
        Return a read-only tuple of this node's children.

        Returns
        --------
        tuple[TreeNode, ...]
            Children in their current ordering.
        """
        return tuple(self._children)

    @property
    def parent(self) -> "TreeNode | None":
        """
        Return the parent of this node, if any.

        Returns
        --------
        TreeNode | None
            The parent node or None if this node does not have a parent.
        """
        return self._parent

    @property
    def root(self) -> "TreeNode":
        """
        Return the root node of the tree containing this node.

        Returns
        --------
        TreeNode
            The root node of the tree.
        """
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def add_child(self, child: "TreeNode") -> "TreeNode":
        """
        Attach a child node to this node.

        This updates both the child's parent pointer and this node's internal
        children list. A child may only have one parent.

        Parameters
        -----------
        child : TreeNode
            The child node to attach.

        Returns
        --------
        TreeNode
            The current node.
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

        Parameters
        -----------
        child : TreeNode
            The child node to remove.
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

        Parameters
        -----------
        parent : TreeNode | None
            The new parent node, or None to detach this node.
        """
        if self.parent is not None:
            self.parent.remove_child(self)
        self._parent = parent
        if parent is not None:
            parent._children.append(self)

    def is_leaf(self) -> bool:
        """
        Determine whether this node is a leaf (has no children).

        Returns
        --------
        bool
            True if the node has no children, otherwise False.
        """
        return not self.children

    def is_internal(self) -> bool:
        """
        Determine whether this node is internal (has one or more children).

        Returns
        --------
        bool
            True if the node has one or more children, otherwise False.
        """
        return not self.is_leaf()

    def get_leaves(self) -> tuple["TreeNode", ...]:
        """
        Return all leaf nodes in the subtree rooted at this node.

        Returns
        --------
        tuple[TreeNode, ...]
            All nodes without children in the subtree.
        """
        return tuple(node for node in self if node.is_leaf())

    def get_internal_nodes(self) -> tuple["TreeNode", ...]:
        """
        Return all internal (non-leaf) nodes in the subtree rooted at this node.

        Returns
        --------
        tuple[TreeNode, ...]
            All nodes that have at least one child in the subtree.
        """
        return tuple(node for node in self if node.is_internal())

    def is_binary(self) -> bool:
        """
        Determine whether the subtree rooted at this node is binary.

        A binary tree has 0 or 2 children at every node.

        Returns
        --------
        bool
            True if every node has 0 or 2 children, otherwise False.
        """
        return all(len(node.children) == 2 for node in self.get_internal_nodes())

    def branch_length_or_raise(self) -> float:
        """
        Return the branch length, raising if it is missing.

        The root node returns 0.0 when its branch length is None to provide a
        convenient origin length for distance calculations.

        Returns
        --------
        float
            The branch length for this node.
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
        """
        Iterate over ancestors from the parent up to (but excluding) a stop node.

        Parameters
        -----------
        stop : TreeNode | None, optional
            If provided, iteration stops before yielding this node. If the
            stop node is not found before reaching the root, a ValueError
            is raised.

        Returns
        --------
        Iterator[TreeNode]
            An iterator over ancestor nodes.
        """
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
        """
        Iterate from this node upward through its ancestors, including self and up to (but excluding) a stop node.

        Parameters
        -----------
        stop : TreeNode | None, optional
            If provided, iteration stops before yielding this node. If the
            stop node is not found before reaching the root, a ValueError
            is raised.

        Returns
        --------
        Iterator[TreeNode]
            An iterator starting at this node and going up the tree.
        """
        if self == stop:
            return
        yield self
        yield from self.iter_ancestors(stop=stop)

    def iter_descendants(self) -> Iterator["TreeNode"]:
        """
        Iterate over all descendants of this node (excluding self).

        Returns
        --------
        Iterator[TreeNode]
            A depth-first iterator over descendant nodes.
        """
        for child in self.children:
            yield child
            yield from child.iter_descendants()

    def iter_preorder(self) -> Iterator["TreeNode"]:
        """
        Iterate over nodes in preorder (self before descendants).

        Returns
        --------
        Iterator[TreeNode]
            Preorder traversal iterator.
        """
        yield self
        yield from self.iter_descendants()

    def iter_inorder(self) -> Iterator["TreeNode"]:
        """
        Iterate over nodes in inorder for binary trees.

        For non-binary trees, a ValueError is raised because inorder is not
        well-defined.

        Returns
        --------
        Iterator[TreeNode]
            Inorder traversal iterator.
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
        """
        Iterate over nodes in postorder (descendants before self).

        Returns
        --------
        Iterator[TreeNode]
            Postorder traversal iterator.
        """
        for child in self.children:
            yield from child.iter_postorder()
        yield self

    def iter_breadth_first(self) -> Iterator["TreeNode"]:
        """
        Iterate over nodes in breadth-first order.

        Returns
        --------
        Iterator[TreeNode]
            Breadth-first traversal iterator.
        """
        queue: deque["TreeNode"] = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    # --------------
    # Paths and hops
    # --------------

    def get_mrca(self, other: "TreeNode") -> "TreeNode":
        """
        Find the most recent common ancestor (MRCA) of two nodes.

        Parameters
        -----------
        other : TreeNode
            The

        Returns
        --------
        TreeNode
            The MRCA of the two nodes.
        """
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

        Parameters
        -----------
        other : TreeNode
            The target node to connect to.

        Returns
        --------
        list[TreeNode]
            The ordered path of nodes between the two inputs.
        """
        mrca = self.get_mrca(other)
        return [
            *self.iter_upward(stop=mrca.parent),
            *reversed(list(other.iter_upward(stop=mrca))),
        ]

    def count_hops(self, other: "TreeNode") -> int:
        """
        Count the number of edges between this node and another node.

        Parameters
        -----------
        other : TreeNode
            The target node to connect to.

        Returns
        --------
        int
            The number of edges (hops) between the two nodes.
        """
        return len(self.get_path(other)) - 1

    def get_distance(self, other: "TreeNode") -> float:
        """
        Compute the branch-length distance between this node and another node.

        Parameters
        -----------
        other : TreeNode
            The target node to connect to.

        Returns
        --------
        float
            The sum of branch lengths along the path between the nodes.
        """
        mrca = self.get_mrca(other)
        path = self.get_path(other)
        path.remove(mrca)
        return sum(node.branch_length_or_raise() for node in path)

    # ---------------
    # Tree properties
    # ---------------

    @property
    def leaf_counts(self) -> dict["TreeNode", int]:
        """
        Compute the number of leaves under each node for the subtree rooted at this node.

        Returns
        --------
        dict[TreeNode, int]
            Mapping from descendant node to the number of leaves under it.
        """
        n_leaves: dict[TreeNode, int] = {}
        for node in self.iter_postorder():
            n_leaves[node] = sum(n_leaves[child] for child in node.children) or 1
        return n_leaves

    @property
    def n_leaves(self) -> int:
        """
        Return the number of leaf nodes for the subtree rooted at this node.

        Returns
        --------
        int
            The count of leaf nodes under this node.
        """
        return len(self.get_leaves())

    @property
    def height_levels(self) -> dict["TreeNode", int]:
        """
        Compute the height level of all nodes for the subtree rooted at this node.

        The height level is defined as the number of edges from a node to its farthest leaf.
        The height level is 0 for leaf nodes, and increases by 1 for each level up the tree.

        Returns
        --------
        dict[TreeNode, int]
            Mapping from descendant node to the distance in edges to its farthest leaf.
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
        The height level is 0 for leaf nodes, and increases by 1 for each level up the tree.

        Returns
        --------
        int
            Distance in edges from the node to its farthest leaf.
        """
        return self.height_levels[self]

    @property
    def heights(self) -> dict["TreeNode", float]:
        """
        Compute the height of all nodes for the subtree rooted at this node.

        The height is defined as the distance in branch-length units from a node to its farthest leaf.
        The height is 0 for leaf nodes, and increases by the branch lengths up the tree.

        Returns
        --------
        dict[TreeNode, float]
            Mapping from descendant node to the distance in branch-length units to its farthest leaf.
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
        The height is 0 for leaf nodes, and increases by the branch lengths up the tree.

        Returns
        --------
        float
            Distance in branch-length units from the node to its farthest leaf.
        """
        return self.heights[self]

    @property
    def depth_levels(self) -> dict["TreeNode", int]:
        """
        Compute the depth level of all nodes for the subtree rooted at this node.

        The depth level is defined as the number of edges from a node to the current node.
        The depth level starts at 0 for the current node, and increases by 1 for each
        level down the tree.

        Returns
        --------
        dict[TreeNode, int]
            Mapping from descendant node to the number of edges to the current node.
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
        The depth level starts at 0 for the root, and increases by 1 for each
        level down the tree.

        Returns
        --------
        int
            The number of edges from the node to the root.
        """
        return self.count_hops(self.root)

    @property
    def depths(self) -> dict["TreeNode", float]:
        """
        Compute the depth of all nodes for the subtree rooted at this node.

        The depth is defined as the distance in branch-length units from a node to the current node.
        The depth starts at 0.0 for the current node, and increases by the branch lengths down the tree.

        Returns
        --------
        dict[TreeNode, float]
            Mapping from descendant node to the distance in branch-length units to the current node.
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

        Returns
        --------
        float
            The distance in branch-length units from this node to the root.
        """
        return self.get_distance(self.root)

    @property
    def times(self) -> dict["TreeNode", float]:
        """
        Compute the time of all nodes for the subtree rooted at this node.

        Times are measured forwards from the current node. They are computed
        by adding the current node's branch length to the relative depth of each node.

        Returns
        --------
        dict[TreeNode, float]
            Mapping from node to its time.
        """
        origin = self.branch_length_or_raise()
        return {node: origin + depth for node, depth in self.depths.items()}

    @property
    def time(self) -> float:
        """
        Compute the time of this node.

        Time is measured forwards from the root. It is computed
        by adding the root's branch length to the depth of the node.

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

        Returns
        --------
        dict[TreeNode, float]
            Mapping from node to its age.
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

        Returns
        --------
        float
            The age of this node.
        """
        return self.root.height - self.get_distance(self.root)

    @property
    def origin(self) -> float:
        """
        Return the origin time of the subtree rooted at this node.

        The origin time is defined as the age of the current node plus its branch length.

        Returns
        --------
        float
            The origin time of the subtree.
        """
        return self.age + self.branch_length_or_raise()

    def compute_sackin_index(self, normalize: bool = False) -> float:
        """
        Compute the Sackin index for the subtree rooted at this node.

        Parameters
        -----------
        normalize : bool, optional
            Whether to normalize the Sackin index to [0, 1] for binary trees. Default is False.

        Returns
        --------
        float
            The Sackin index, normalized if requested.
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

        Returns
        --------
        TreeNode
            A deep copy of this node and its subtree.
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
            Mapping used to sort children. Defaults to the number of leaf descendants if None.
        """
        if key is None:
            key = self.leaf_counts

        for node in self:
            node._children.sort(key=lambda child: key[child])

    def get_descendant(self, name: str) -> "TreeNode":
        """
        Find the first descendant node with the given name.

        Parameters
        -----------
        name : str
            The node name to search for.

        Returns
        --------
        TreeNode
            The first node whose name matches.
        """
        for node in self:
            if node.name == name:
                return node
        raise ValueError(f"Node {name} not found.")

    # -------------
    # Dunder methods
    # -------------

    def __repr__(self) -> str:
        """
        Return a string representation of the node.

        Returns
        --------
        str
            String representation of the node.
        """
        return f"TreeNode(name='{self.name}', branch_length={self.branch_length}, metadata={self.metadata})"

    def __iter__(self) -> Iterator["TreeNode"]:
        """
        Iterate over all nodes in the subtree rooted at this node in preorder.

        Returns
        --------
        Iterator[TreeNode]
            Preorder traversal iterator.
        """
        return self.iter_preorder()

    def __len__(self) -> int:
        """
        Return the number of nodes in the subtree rooted at this node.

        Returns
        --------
        int
            The count of nodes in the subtree.
        """
        return sum(1 for _ in self)
