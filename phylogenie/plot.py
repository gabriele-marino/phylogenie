from enum import Enum
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # pyright: ignore

from phylogenie.tree import Tree
from phylogenie.utils import get_node_depth_levels, get_node_depths


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
    show_hist: bool = True,
    hist_kwargs: dict[str, Any] | None = None,
    hist_axes_kwargs: dict[str, Any] | None = None,
) -> Axes | tuple[Axes, Axes]:
    if ax is None:
        ax = plt.gca()

    xs = (
        get_node_depth_levels(tree)
        if any(node.branch_length is None for node in tree)
        else get_node_depths(tree)
    )
    ys = {node: i for i, node in enumerate(tree.inorder_traversal())}

    if color_by is not None:
        features = [node.get(color_by) for node in tree if color_by in node.features]

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
                f: mcolors.to_hex(colormap(i)) for i, f in enumerate(set(features))
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
                    for f in feature_colors
                ]
                if any(color_by not in node.features for node in tree):
                    legend_handles.append(
                        mpatches.Patch(color=default_color, label="NA")
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

            if show_hist:
                default_hist_axes_kwargs = {"width": "25%", "height": "25%"}
                if hist_axes_kwargs is not None:
                    default_hist_axes_kwargs.update(hist_axes_kwargs)
                hist_ax = inset_axes(ax, **default_hist_axes_kwargs)  # pyright: ignore

                hist_kwargs = {} if hist_kwargs is None else hist_kwargs
                _, bins, patches = hist_ax.hist(  # pyright: ignore
                    features, **hist_kwargs
                )

                for patch, b0, b1 in zip(  # pyright: ignore
                    patches, bins[:-1], bins[1:]  # pyright: ignore
                ):
                    midpoint = (b0 + b1) / 2  # pyright: ignore
                    patch.set_facecolor(colormap(norm(midpoint)))  # pyright: ignore
            else:
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
