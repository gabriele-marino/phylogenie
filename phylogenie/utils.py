from itertools import chain
from math import comb

from phylogenie.tree import Tree


def get_node_leaf_counts(tree: Tree) -> dict[Tree, int]:
    n_leaves: dict[Tree, int] = {}
    for node in tree.postorder_traversal():
        n_leaves[node] = sum(n_leaves[child] for child in node.children) or 1
    return n_leaves


def get_node_depth_levels(tree: Tree) -> dict[Tree, int]:
    depth_levels: dict[Tree, int] = {}
    for node in tree:
        if node.parent is None:
            depth_levels[node] = 0
        else:
            depth_levels[node] = depth_levels[node.parent] + 1
    return depth_levels


def get_node_depths(tree: Tree) -> dict[Tree, float]:
    depths: dict[Tree, float] = {}
    for node in tree:
        if node.parent is None:
            depths[node] = 0 if node.branch_length is None else node.branch_length
        else:
            if node.branch_length is None:
                raise ValueError(f"Branch length of node {node.name} is not set.")
            depths[node] = depths[node.parent] + node.branch_length
    return depths


def get_node_height_levels(tree: Tree) -> dict[Tree, int]:
    height_levels: dict[Tree, int] = {}
    for node in tree.postorder_traversal():
        if node.is_leaf():
            height_levels[node] = 0
        else:
            height_levels[node] = max(
                1 + height_levels[child] for child in node.children
            )
    return height_levels


def get_node_heights(tree: Tree) -> dict[Tree, float]:
    heights: dict[Tree, float] = {}
    for node in tree.postorder_traversal():
        if node.is_leaf():
            heights[node] = 0
        else:
            if any(child.branch_length is None for child in node.children):
                raise ValueError(
                    f"Branch length of one or more children of node {node.name} is not set."
                )
            heights[node] = max(
                child.branch_length + heights[child]  # pyright: ignore
                for child in node.children
            )
    return heights


def get_mrca(node1: Tree, node2: Tree) -> Tree:
    node1_ancestors = set(node1.iter_ancestors())
    for node2_ancestor in node2.iter_ancestors():
        if node2_ancestor in node1_ancestors:
            return node2_ancestor
    raise ValueError(f"No common ancestor found between node {node1} and node {node2}.")


def get_distance(node1: Tree, node2: Tree) -> float:
    mrca = get_mrca(node1, node2)
    path = list(chain(node1.iter_ancestors(stop=mrca), node2.iter_ancestors(stop=mrca)))
    if any(node.branch_length is None for node in path):
        return len(path)
    return sum(node.branch_length for node in path)  # pyright: ignore


def compute_sackin_index(tree: Tree, normalize: bool = False) -> float:
    """
    Compute the Sackin index of a tree.

    Parameters
    ----------
    tree : Tree
        The input tree.
    normalize : bool, optional
        Whether to normalize the index between 0 and 1, by default False.

    Returns
    -------
    float
        The Sackin index of the tree.

    References
    ----------
    - Kwang-Tsao Shao and Robert R Sokal. Tree Balance. Systematic Zoology, 39(3):266, 1990.
    """
    leaves = tree.get_leaves()
    depth_levels = get_node_depth_levels(tree)
    sackin_index = sum(depth_levels[leaf] for leaf in leaves)
    if normalize:
        n = len(leaves)
        max_sackin_index = (n + 2) * (n - 1) / 2
        return (sackin_index - n) / (max_sackin_index - n)
    return sackin_index


def compute_colless_index(tree: Tree, normalize: bool = False) -> float:
    """
    Compute the Colless index of a binary tree.

    Parameters
    ----------
    tree : Tree
        The input binary tree.
    normalize : bool, optional
        Whether to normalize the index between 0 and 1, by default False.

    Returns
    -------
    float
        The Colless index of the tree.

    References
    ----------
    - Donald H. Colless. Review of phylogenetics: the theory and practice of phylogenetic systematics. Systematic Zoology, 31(1):100–104, 1982.
    """
    if not tree.is_binary():
        raise ValueError("Colless index is only defined for binary trees.")

    internal_nodes = tree.get_internal_nodes()
    if not internal_nodes:
        raise ValueError(
            "Tree must have at least one internal node to compute the Colless index."
        )

    colless_index = 0
    leaf_counts = get_node_leaf_counts(tree)
    for node in internal_nodes:
        left, right = node.children
        colless_index += abs(leaf_counts[left] - leaf_counts[right])
    if normalize:
        n_leaves = len(leaf_counts) - len(internal_nodes)
        max_colless_index = comb(n_leaves, 2)
        return colless_index / max_colless_index
    return colless_index


def compute_mean_leaf_pairwise_distance(tree: Tree) -> float:
    """
    Compute the mean pairwise distance between all pairs of leaves in the tree.

    Parameters
    ----------
    tree : Tree
        The input tree.

    Returns
    -------
    float
        The mean pairwise distance between all pairs of leaves in the tree.
    """
    leaves = tree.get_leaves()
    n_leaves = len(leaves)
    if n_leaves < 2:
        return 0.0

    total_distance = sum(
        get_distance(leaves[i], leaves[j])
        for i in range(n_leaves)
        for j in range(i + 1, n_leaves)
    )
    return total_distance / comb(n_leaves, 2)
