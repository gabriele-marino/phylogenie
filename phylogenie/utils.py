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
            depths[node] = 0
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
