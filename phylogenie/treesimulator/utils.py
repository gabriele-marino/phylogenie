import math
from math import comb

from phylogenie.treesimulator.tree import Tree


def get_node_leaf_counts(tree: Tree) -> dict[Tree, int]:
    n_leaves: dict[Tree, int] = {}
    for node in tree.postorder_traversal():
        n_leaves[node] = sum(n_leaves[child] for child in node.children) or 1
    return n_leaves


def get_node_depth_levels(tree: Tree) -> dict[Tree, int]:
    depth_levels: dict[Tree, int] = {tree: tree.depth_level}
    for node in tree.iter_descendants():
        depth_levels[node] = depth_levels[node.parent] + 1  # pyright: ignore
    return depth_levels


def get_node_depths(tree: Tree) -> dict[Tree, float]:
    depths: dict[Tree, float] = {tree: tree.depth}
    for node in tree.iter_descendants():
        parent_depth = depths[node.parent]  # pyright: ignore
        depths[node] = node.branch_length_or_raise() + parent_depth
    return depths


def get_node_height_levels(tree: Tree) -> dict[Tree, int]:
    height_levels: dict[Tree, int] = {}
    for node in tree.postorder_traversal():
        height_levels[node] = (
            0
            if node.is_leaf()
            else max(1 + height_levels[child] for child in node.children)
        )
    return height_levels


def get_node_heights(tree: Tree) -> dict[Tree, float]:
    heights: dict[Tree, float] = {}
    for node in tree.postorder_traversal():
        heights[node] = (
            0
            if node.is_leaf()
            else max(
                child.branch_length_or_raise() + heights[child]
                for child in node.children
            )
        )
    return heights


def get_mrca(node1: Tree, node2: Tree) -> Tree:
    node1_ancestors = set(node1.iter_upward())
    for node2_ancestor in node2.iter_upward():
        if node2_ancestor in node1_ancestors:
            return node2_ancestor
    raise ValueError(f"No common ancestor found between node {node1} and node {node2}.")


def get_path(node1: Tree, node2: Tree) -> list[Tree]:
    mrca = get_mrca(node1, node2)
    return [
        *node1.iter_upward(stop=mrca.parent),
        *reversed(list(node2.iter_upward(stop=mrca))),
    ]


def count_hops(node1: Tree, node2: Tree) -> int:
    return len(get_path(node1, node2)) - 1


def get_distance(node1: Tree, node2: Tree) -> float:
    mrca = get_mrca(node1, node2)
    path = get_path(node1, node2)
    path.remove(mrca)
    return sum(node.branch_length_or_raise() for node in path)


def compute_sackin_index(tree: Tree, normalize: bool = False) -> float:
    depth_levels = get_node_depth_levels(tree)
    sackin_index = sum(dl for node, dl in depth_levels.items() if node.is_leaf())
    if normalize:
        if not tree.is_binary():
            raise ValueError(
                "Normalized Sackin index is only defined for binary trees."
            )
        n = tree.n_leaves
        h = math.floor(math.log2(n))
        min_sackin_index = n * (h + 2) - 2 ** (h + 1)
        max_sackin_index = n * (n - 1) / 2
        return (sackin_index - min_sackin_index) / (max_sackin_index - min_sackin_index)
    return sackin_index


def compute_mean_leaf_pairwise_distance(tree: Tree) -> float:
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
