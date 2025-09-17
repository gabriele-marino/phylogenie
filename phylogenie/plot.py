from enum import Enum

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from phylogenie.tree import Tree
from phylogenie.utils import get_node_depths


class Coloring(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


def plot_tree(
    tree: Tree,
    ax: Axes | None = None,
    color_by: str | None = None,
    default_color: str = "black",
    coloring: str | Coloring | None = None,
    cmap: str | None = None,
    show_legend: bool = True,
    x_feature: str | None = None,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    xs = (
        get_node_depths(tree)
        if x_feature is None
        else {node: node.get(x_feature) for node in tree}
    )
    ys = {node: i for i, node in enumerate(tree.inorder_traversal())}

    if color_by is not None:
        features = set(node.get(color_by) for node in tree if color_by in node.features)
        if coloring is None and any(isinstance(f, float) for f in features):
            coloring = Coloring.CONTINUOUS
        elif coloring is None:
            coloring = Coloring.DISCRETE

        if coloring == Coloring.DISCRETE:
            if any(isinstance(f, float) for f in features):
                raise ValueError(
                    "Discrete coloring selected but feature values are not all categorical."
                )
            colormap = plt.get_cmap("tab20" if cmap is None else cmap)
            feature_colors = {
                f: mcolors.to_hex(colormap(i)) for i, f in enumerate(features)
            }
            colors = {
                node: (
                    feature_colors[node.get(color_by)]
                    if color_by in node.features
                    else default_color
                )
                for node in tree
            }

            if show_legend:
                legend_handles = [
                    mpatches.Patch(color=feature_colors[f], label=str(f))
                    for f in features
                ]
                if any(color_by not in node.features for node in tree):
                    legend_handles.append(
                        mpatches.Patch(color=default_color, label="None")
                    )
                ax.legend(handles=legend_handles, title=color_by)  # pyright: ignore

        elif coloring == Coloring.CONTINUOUS:
            norm = mcolors.Normalize(vmin=min(features), vmax=max(features))
            colormap = plt.get_cmap("viridis" if cmap is None else cmap)
            colors = {
                node: (
                    colormap(norm(float(node.get(color_by))))
                    if color_by in node.features
                    else default_color
                )
                for node in tree
            }

            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            ax.get_figure().colorbar(sm, ax=ax)  # pyright: ignore

        else:
            raise ValueError(
                f"Unknown coloring method: {coloring}. Choices are {list(Coloring)}."
            )
    else:
        colors = {node: default_color for node in tree}

    for node in tree:
        x1, y1 = xs[node], ys[node]
        if node.parent is None:
            ax.hlines(y=y1, xmin=0, xmax=x1, color=colors[node])  # pyright: ignore
            continue
        x0, y0 = xs[node.parent], ys[node.parent]
        ax.vlines(x=x0, ymin=y0, ymax=y1, color=colors[node])  # pyright: ignore
        ax.hlines(y=y1, xmin=x0, xmax=x1, color=colors[node])  # pyright: ignore

    ax.set_yticks([])  # pyright: ignore
    return ax
