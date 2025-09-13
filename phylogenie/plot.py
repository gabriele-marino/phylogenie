from enum import Enum

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from phylogenie import Tree
from phylogenie.tree import Tree
from phylogenie.utils import get_times


class Coloring(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


def plot_tree(
    tree: Tree,
    ax: plt.Axes | None = None,  # pyright: ignore
    color_by: str | None = None,
    default_color: str = "black",
    coloring: str | Coloring | None = None,
    cmap: str | None = None,
    show_legend: bool = True,
) -> plt.Axes:  # pyright: ignore
    if ax is None:
        ax = plt.gca()

    xs = get_times(tree)
    ys = {node: i for i, node in enumerate(tree.inorder_traversal())}

    if color_by is not None:
        features = set(node.get(color_by) for node in tree)
        if coloring is None and any(isinstance(f, float) for f in features):
            coloring = Coloring.CONTINUOUS
        elif coloring is None:
            coloring = Coloring.DISCRETE
        if coloring == Coloring.DISCRETE:
            if any(isinstance(f, float) for f in features):
                raise ValueError(
                    "Discrete coloring selected but feature values are not all categorical."
                )
            cmap = "tab20" if cmap is None else cmap
            colormap = plt.get_cmap(cmap, len(features))
            feature_colors = {
                f: mcolors.to_hex(colormap(i)) for i, f in enumerate(features)
            }
            colors = {node: feature_colors[node.get(color_by)] for node in tree}
            legend_handles = [
                mpatches.Patch(color=feature_colors[f], label=str(f)) for f in features
            ]
            if show_legend:
                ax.legend(handles=legend_handles, title=color_by)  # pyright: ignore
        elif coloring in {Coloring.CONTINUOUS}:
            cmap = "viridis" if cmap is None else cmap
            values = list(map(float, features))
            norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
            colormap = plt.get_cmap(cmap)
            colors = {node: colormap(norm(float(node.get(color_by)))) for node in tree}
        else:
            raise ValueError(
                f"Unknown coloring method: {coloring}. Choices are {list(Coloring)}."
            )
    else:
        colors = {node: default_color for node in tree}

    for node in tree:
        x1, y1 = xs[node.name], ys[node]
        if node.parent is None:
            ax.hlines(y=y1, xmin=0, xmax=x1, color=colors[node])  # pyright: ignore
            continue
        x0, y0 = xs[node.parent.name], ys[node.parent]
        ax.vlines(x=x0, ymin=y0, ymax=y1, color=colors[node])  # pyright: ignore
        ax.hlines(y=y1, xmin=x0, xmax=x1, color=colors[node])  # pyright: ignore

    ax.set_yticks([])  # pyright: ignore
    return ax
