from phylogenie.tree import Tree


def get_n_tips(tree: Tree) -> dict[str, int]:
    n_tips: dict[str, int] = {}
    for node in tree.postorder_traversal():
        n_tips[node.name] = (
            1 if node.is_leaf() else sum(n_tips[child.name] for child in node.children)
        )
    return n_tips


def get_times(tree: Tree) -> dict[str, float]:
    times: dict[str, float] = {}
    for node in tree:
        parent_time = 0 if node.parent is None else times[node.parent.name]
        times[node.name] = node.parse_branch_length() + parent_time
    return times


def get_heights(tree: Tree) -> dict[str, int]:
    heights: dict[str, int] = {}
    for node in tree.postorder_traversal():
        if node.is_leaf():
            heights[node.name] = 0
        else:
            heights[node.name] = 1 + max(heights[child.name] for child in node.children)
    return heights
