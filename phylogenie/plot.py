import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from phylogenie import Tree
from phylogenie.tree import Tree
from phylogenie.utils import get_times


def plot_tree(
    tree: Tree,
    ax: plt.Axes | None = None,  # pyright: ignore
    color_by: str | None = None,
    default_color: str = "black",
    cmap: str = "tab20",
) -> plt.Axes:  # pyright: ignore
    if ax is None:
        ax = plt.gca()

    xs = get_times(tree)
    ys = {node.name: i for i, node in enumerate(tree.inorder_traversal())}
    if color_by is not None:
        features = set(node.get(color_by) for node in tree)
        feature_colors = {
            f: mcolors.to_hex(plt.get_cmap(cmap, len(features))(i))
            for i, f in enumerate(features)
        }
        colors = {node.name: feature_colors[node.get(color_by)] for node in tree}
    else:
        colors = {node.name: default_color for node in tree}

    for node in tree:
        if node.parent is None:
            continue
        x0, y0 = xs[node.parent.name], ys[node.parent.name]
        x1, y1 = xs[node.name], ys[node.name]
        ax.plot([x0, x0], [y0, y1], color=colors[node.name])  # pyright: ignore
        ax.plot([x0, x1], [y1, y1], color=colors[node.name])  # pyright: ignore
    ax.set_yticks([])  # pyright: ignore
    return ax
