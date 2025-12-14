from enum import Enum
from typing import Any, Callable

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # pyright: ignore

from phylogenie.treesimulator import Tree, get_node_depth_levels, get_node_depths


class Coloring(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


Color = str | tuple[float, float, float] | tuple[float, float, float, float]


def draw_colored_tree(
    tree: Tree, ax: Axes | None = None, colors: Color | dict[Tree, Color] = "black"
) -> Axes:
    if ax is None:
        ax = plt.gca()

    if not isinstance(colors, dict):
        colors = {node: colors for node in tree}

    xs = (
        get_node_depth_levels(tree)
        if any(node.branch_length is None for node in tree.iter_descendants())
        else get_node_depths(tree)
    )
    ys: dict[Tree, float] = {node: i for i, node in enumerate(tree.get_leaves())}
    for node in tree.postorder_traversal():
        if node.is_internal():
            ys[node] = sum(ys[child] for child in node.children) / len(node.children)

    if tree.branch_length is not None:
        ax.hlines(y=ys[tree], xmin=0, xmax=xs[tree], color=colors[tree])  # pyright: ignore
    for node in tree:
        x1, y1 = xs[node], ys[node]
        for child in node.children:
            x2, y2 = xs[child], ys[child]
            ax.hlines(y=y2, xmin=x1, xmax=x2, color=colors[child])  # pyright: ignore
            ax.vlines(x=x1, ymin=y1, ymax=y2, color=colors[child])  # pyright: ignore

    ax.set_yticks([])  # pyright: ignore
    return ax


def draw_tree(
    tree: Tree,
    ax: Axes | None = None,
    color_by: str | dict[str, Any] | None = None,
    coloring: str | Coloring | None = None,
    default_color: Color = "black",
    colormap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_legend: bool = True,
    labels: dict[Any, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    show_hist: bool = True,
    hist_kwargs: dict[str, Any] | None = None,
    hist_axes_kwargs: dict[str, Any] | None = None,
) -> Axes | tuple[Axes, Axes]:
    if ax is None:
        ax = plt.gca()

    if color_by is None:
        return draw_colored_tree(tree, ax, colors=default_color)

    if isinstance(color_by, str):
        features = {node: node[color_by] for node in tree if color_by in node.metadata}
    else:
        features = {node: color_by[node.name] for node in tree if node.name in color_by}
    values = list(features.values())

    if coloring is None:
        coloring = (
            Coloring.CONTINUOUS
            if any(isinstance(f, float) for f in values)
            else Coloring.DISCRETE
        )
    if colormap is None:
        colormap = "tab20" if coloring == Coloring.DISCRETE else "viridis"
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)

    def _get_colors(feature_map: Callable[[Any], Color]) -> dict[Tree, Color]:
        return {
            node: feature_map(features[node]) if node in features else default_color
            for node in tree
        }

    if coloring == Coloring.DISCRETE:
        if any(isinstance(f, float) for f in values):
            raise ValueError(
                "Discrete coloring selected but feature values are not all categorical."
            )
        feature_colors = {
            f: mcolors.to_hex(colormap(i)) for i, f in enumerate(set(values))
        }
        colors = _get_colors(lambda f: feature_colors[f])

        if show_legend:
            legend_handles = [
                mpatches.Patch(
                    color=feature_colors[f],
                    label=str(f) if labels is None else labels[f],
                )
                for f in feature_colors
            ]
            if any(color_by not in node.metadata for node in tree):
                legend_handles.append(mpatches.Patch(color=default_color, label="NA"))
            if legend_kwargs is None:
                legend_kwargs = {}
            ax.legend(handles=legend_handles, **legend_kwargs)  # pyright: ignore

        return draw_colored_tree(tree, ax, colors)

    if coloring == Coloring.CONTINUOUS:
        vmin = min(values) if vmin is None else vmin
        vmax = max(values) if vmax is None else vmax
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colors = _get_colors(lambda f: colormap(norm(float(f))))

        if show_hist:
            default_hist_axes_kwargs = {"width": "25%", "height": "25%"}
            if hist_axes_kwargs is not None:
                default_hist_axes_kwargs.update(hist_axes_kwargs)
            hist_ax = inset_axes(ax, **default_hist_axes_kwargs)  # pyright: ignore

            hist_kwargs = {} if hist_kwargs is None else hist_kwargs
            _, bins, patches = hist_ax.hist(values, **hist_kwargs)  # pyright: ignore

            for patch, b0, b1 in zip(patches, bins[:-1], bins[1:]):  # pyright: ignore
                midpoint = (b0 + b1) / 2  # pyright: ignore
                patch.set_facecolor(colormap(norm(midpoint)))  # pyright: ignore
            return draw_colored_tree(tree, ax, colors), hist_ax  # pyright: ignore

        else:
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            ax.get_figure().colorbar(sm, ax=ax)  # pyright: ignore
            return draw_colored_tree(tree, ax, colors)

    raise ValueError(
        f"Unknown coloring method: {coloring}. Choices are {list(Coloring)}."
    )
