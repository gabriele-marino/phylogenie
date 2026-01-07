import datetime
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, overload

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # pyright: ignore

from phylogenie.treesimulator import (
    Tree,
    get_node_ages,
    get_node_depth_levels,
    get_node_depths,
)


@dataclass
class CalibrationNode:
    node: Tree
    date: datetime.date


Color = str | tuple[float, float, float] | tuple[float, float, float, float]


def draw_tree(
    tree: Tree,
    ax: Axes | None = None,
    colors: Color | Mapping[Tree, Color] = "black",
    backward_time: bool = False,
    branch_kwargs: dict[str, Any] | None = None,
    sampled_ancestor_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """
    Draw a phylogenetic tree with colored branches.

    Parameters
    ----------
    tree : Tree
        The phylogenetic tree to draw.
    ax : Axes | None, optional
        The matplotlib Axes to draw on. If None, uses the current Axes.
    colors : Color | Mapping[Tree, Color], optional
        A single color for all branches or a mapping from each node to a color.
    backward_time : bool, optional
        If True, the x-axis is inverted to represent time going backward.
    branch_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the branch drawing functions.
    sampled_ancestor_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to highlight sampled ancestors.

    Returns
    -------
    Axes
        The Axes with the drawn tree.
    """
    if ax is None:
        ax = plt.gca()
    if branch_kwargs is None:
        branch_kwargs = {}
    if sampled_ancestor_kwargs is None:
        sampled_ancestor_kwargs = {}
    if "marker" not in sampled_ancestor_kwargs:
        sampled_ancestor_kwargs["marker"] = "o"

    if not isinstance(colors, Mapping):
        colors = {node: colors for node in tree}

    xs = (
        get_node_ages(tree)
        if backward_time
        else get_node_depth_levels(tree)
        if any(node.branch_length is None for node in tree.iter_descendants())
        else get_node_depths(tree)
    )

    leaves = tree.get_leaves()
    ys: dict[Tree, float] = {
        node: i
        for i, node in enumerate(leaves)
        if node.parent is None or node.branch_length != 0
    }
    for node in tree.postorder_traversal():
        if node.is_internal():
            children = [child for child in node.children if child.branch_length != 0]
            ys[node] = sum(ys[child] for child in children) / len(children)
    for leaf in leaves:
        if leaf.parent is not None and leaf.branch_length == 0:
            ys[leaf] = ys[leaf.parent]

    if tree.branch_length is not None:
        xmin = xs[tree] + tree.branch_length if backward_time else 0
        ax.hlines(  # pyright: ignore
            y=ys[tree], xmin=xmin, xmax=xs[tree], color=colors[tree], **branch_kwargs
        )
    for node in tree:
        x1, y1 = xs[node], ys[node]
        if node.parent is not None and node.branch_length == 0:
            ax.plot(x1, y1, color=colors[node], **sampled_ancestor_kwargs)  # pyright: ignore
        for child in node.children:
            x2, y2 = xs[child], ys[child]
            ax.hlines(y=y2, xmin=x1, xmax=x2, color=colors[child], **branch_kwargs)  # pyright: ignore
            ax.vlines(x=x1, ymin=y1, ymax=y2, color=colors[child], **branch_kwargs)  # pyright: ignore

    if backward_time:
        ax.invert_xaxis()

    ax.set_yticks([])  # pyright: ignore
    return ax


def _depth_to_date(
    depth: float, calibration_nodes: tuple[CalibrationNode, CalibrationNode]
) -> datetime.date:
    """
    Convert a depth value to a date using linear interpolation between two calibration nodes.

    Parameters
    ----------
    depth : float
        The depth value to convert.
    calibration_nodes : tuple[CalibrationNode, CalibrationNode]
        Two calibration nodes defining the mapping from depth to date.

    Returns
    -------
    datetime.date
        The interpolated date corresponding to the given depth.
    """
    node1, node2 = calibration_nodes
    depth1, depth2 = node1.node.depth, node2.node.depth
    date1, date2 = node1.date, node2.date
    return date1 + (depth - depth1) * (date2 - date1) / (depth2 - depth1)


def draw_dated_tree(
    tree: Tree,
    calibration_nodes: tuple[CalibrationNode, CalibrationNode],
    ax: Axes | None = None,
    colors: Color | Mapping[Tree, Color] = "black",
    branch_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """
    Draw a phylogenetic tree with branches positioned according to calibrated dates.

    Parameters
    ----------
    tree : Tree
        The phylogenetic tree to draw.
    calibration_nodes : tuple[CalibrationNode, CalibrationNode]
        Two calibration nodes defining the mapping from depth to date.
    ax : Axes | None, optional
        The matplotlib Axes to draw on. If None, uses the current Axes.
    colors : Color | Mapping[Tree, Color], optional
        A single color for all branches or a mapping from each node to a color.
    branch_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the branch drawing functions.

    Returns
    -------
    Axes
        The Axes with the drawn dated tree.
    """
    if ax is None:
        ax = plt.gca()
    if branch_kwargs is None:
        branch_kwargs = {}

    if not isinstance(colors, Mapping):
        colors = {node: colors for node in tree}

    xs = {
        node: _depth_to_date(depth=depth, calibration_nodes=calibration_nodes)
        for node, depth in get_node_depths(tree).items()
    }

    ys: dict[Tree, float] = {node: i for i, node in enumerate(tree.get_leaves())}
    for node in tree.postorder_traversal():
        if node.is_internal():
            ys[node] = sum(ys[child] for child in node.children) / len(node.children)

    if tree.branch_length is not None:
        origin_date = _depth_to_date(depth=0, calibration_nodes=calibration_nodes)
        ax.hlines(  # pyright: ignore
            y=ys[tree],
            xmin=mdates.date2num(origin_date),  # pyright: ignore
            xmax=mdates.date2num(xs[tree]),  # pyright: ignore
            color=colors[tree],
            **branch_kwargs,
        )
    for node in tree:
        x1, y1 = xs[node], ys[node]
        for child in node.children:
            x2, y2 = xs[child], ys[child]
            ax.hlines(  # pyright: ignore
                y=y2,
                xmin=mdates.date2num(x1),  # pyright: ignore
                xmax=mdates.date2num(x2),  # pyright: ignore
                color=colors[child],
                **branch_kwargs,
            )
            ax.vlines(  # pyright: ignore
                x=mdates.date2num(x1),  # pyright: ignore
                ymin=y1,
                ymax=y2,
                color=colors[child],
                **branch_kwargs,
            )

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", labelrotation=45)  # pyright: ignore

    ax.set_yticks([])  # pyright: ignore
    return ax


def _init_colored_tree_categorical(
    tree: Tree,
    color_by: str,
    ax: Axes | None = None,
    default_color: Color = "black",
    colormap: str | Mapping[str, Color] | Colormap = "tab20",
    show_legend: bool = True,
    labels: Mapping[Any, str] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
) -> tuple[Axes, dict[Tree, Color]]:
    """
    Initialize colors for drawing a tree based on categorical metadata.

    Parameters
    ----------
    tree : Tree
        The phylogenetic tree.
    color_by : str
        The metadata key to color branches by.
    ax : Axes | None, optional
        The matplotlib Axes to draw on. If None, uses the current Axes.
    default_color : Color, optional
        The color to use for nodes without the specified metadata.
    colormap : str | Mapping[str, Color] | Colormap, optional
        The colormap to use for coloring categories.
        If a string, it is used to get a matplotlib colormap.
        If a mapping, it maps category values to colors directly.
        Defaults to 'tab20'.
    show_legend : bool, optional
        Whether to display a legend for the categories.
    labels : Mapping[Any, str] | None, optional
        A mapping from category values to labels for the legend.
    legend_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the legend.

    Returns
    -------
    tuple[Axes, dict[Tree, Color]]
        The Axes and a dictionary mapping each node to its assigned color.
    """
    if ax is None:
        ax = plt.gca()

    features = {node: node[color_by] for node in tree if color_by in node.metadata}
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    if isinstance(colormap, Colormap):
        feature_colors = {
            f: mcolors.to_hex(colormap(i)) for i, f in enumerate(set(features.values()))
        }
    else:
        feature_colors = colormap

    colors = {
        node: feature_colors[features[node]] if node in features else default_color
        for node in tree
    }

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

    return ax, colors


def draw_colored_tree_categorical(
    tree: Tree,
    color_by: str,
    ax: Axes | None = None,
    backward_time: bool = False,
    default_color: Color = "black",
    colormap: str | Mapping[str, Color] | Colormap = "tab20",
    show_legend: bool = True,
    labels: Mapping[Any, str] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    branch_kwargs: dict[str, Any] | None = None,
    sampled_ancestor_kwargs: dict[str, Any] | None = None,
):
    """
    Draw a phylogenetic tree with branches colored based on categorical metadata.

    Parameters
    ----------
    tree : Tree
        The phylogenetic tree to draw.
    color_by : str
        The metadata key to color branches by.
    ax : Axes | None, optional
        The matplotlib Axes to draw on. If None, uses the current Axes.
    backward_time : bool, optional
        If True, the x-axis is inverted to represent time going backward.
    default_color : Color, optional
        The color to use for nodes without the specified metadata.
    colormap : str | Mapping[str, Color] | Colormap, optional
        The colormap to use for coloring categories.
        If a string, it is used to get a matplotlib colormap.
        If a mapping, it maps category values to colors directly.
        Defaults to 'tab20'.
    show_legend : bool, optional
        Whether to display a legend for the categories.
    labels : Mapping[Any, str] | None, optional
        A mapping from category values to labels for the legend.
    legend_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the legend.
    branch_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the branch drawing functions.
    sampled_ancestor_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to highlight sampled ancestors.

    Returns
    -------
    Axes
        The Axes with the drawn colored tree.
    """
    ax, colors = _init_colored_tree_categorical(
        tree=tree,
        color_by=color_by,
        ax=ax,
        default_color=default_color,
        colormap=colormap,
        show_legend=show_legend,
        labels=labels,
        legend_kwargs=legend_kwargs,
    )
    return draw_tree(
        tree=tree,
        ax=ax,
        colors=colors,
        backward_time=backward_time,
        branch_kwargs=branch_kwargs,
        sampled_ancestor_kwargs=sampled_ancestor_kwargs,
    )


def draw_colored_dated_tree_categorical(
    tree: Tree,
    calibration_nodes: tuple[CalibrationNode, CalibrationNode],
    color_by: str,
    ax: Axes | None = None,
    default_color: Color = "black",
    colormap: str | Mapping[str, Color] | Colormap = "tab20",
    show_legend: bool = True,
    labels: Mapping[Any, str] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    branch_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """
    Draw a dated phylogenetic tree with branches colored based on categorical metadata.

    Parameters
    ----------
    tree : Tree
        The phylogenetic tree to draw.
    calibration_nodes : tuple[CalibrationNode, CalibrationNode]
        Two calibration nodes defining the mapping from depth to date.
    color_by : str
        The metadata key to color branches by.
    ax : Axes | None, optional
        The matplotlib Axes to draw on. If None, uses the current Axes.
    default_color : Color, optional
        The color to use for nodes without the specified metadata.
    colormap : str | Mapping[str, Color] | Colormap, optional
        The colormap to use for coloring categories.
        If a string, it is used to get a matplotlib colormap.
        If a mapping, it maps category values to colors directly.
        Defaults to 'tab20'.
    show_legend : bool, optional
        Whether to display a legend for the categories.
    labels : Mapping[Any, str] | None, optional
        A mapping from category values to labels for the legend.
    legend_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the legend.
    branch_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the branch drawing functions.

    Returns
    -------
    Axes
        The Axes with the drawn colored dated tree.
    """
    ax, colors = _init_colored_tree_categorical(
        tree=tree,
        color_by=color_by,
        ax=ax,
        default_color=default_color,
        colormap=colormap,
        show_legend=show_legend,
        labels=labels,
        legend_kwargs=legend_kwargs,
    )
    return draw_dated_tree(
        tree=tree,
        calibration_nodes=calibration_nodes,
        ax=ax,
        colors=colors,
        branch_kwargs=branch_kwargs,
    )


@overload
def _init_colored_tree_continuous(
    tree: Tree,
    color_by: str,
    ax: Axes | None = ...,
    default_color: Color = ...,
    colormap: str | Colormap = ...,
    vmin: float | None = ...,
    vmax: float | None = ...,
    *,
    show_hist: Literal[False],
    hist_kwargs: dict[str, Any] | None = ...,
    hist_axes_kwargs: dict[str, Any] | None = ...,
) -> tuple[Axes, dict[Tree, Color]]: ...
@overload
def _init_colored_tree_continuous(
    tree: Tree,
    color_by: str,
    ax: Axes | None = ...,
    default_color: Color = ...,
    colormap: str | Colormap = ...,
    vmin: float | None = ...,
    vmax: float | None = ...,
    *,
    show_hist: Literal[True] = True,
    hist_kwargs: dict[str, Any] | None = ...,
    hist_axes_kwargs: dict[str, Any] | None = ...,
) -> tuple[Axes, dict[Tree, Color], Axes]: ...
def _init_colored_tree_continuous(
    tree: Tree,
    color_by: str,
    ax: Axes | None = None,
    default_color: Color = "black",
    colormap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    show_hist: bool = True,
    hist_kwargs: dict[str, Any] | None = None,
    hist_axes_kwargs: dict[str, Any] | None = None,
) -> tuple[Axes, dict[Tree, Color]] | tuple[Axes, dict[Tree, Color], Axes]:
    """
    Initialize colors for drawing a tree based on continuous metadata.

    Parameters
    ----------
    tree : Tree
        The phylogenetic tree.
    color_by : str
        The metadata key to color branches by.
    ax : Axes | None, optional
        The matplotlib Axes to draw on. If None, uses the current Axes.
    default_color : Color, optional
        The color to use for nodes without the specified metadata.
    colormap : str | Colormap, optional
        The colormap to use for coloring continuous values. Defaults to 'viridis'.
    vmin : float | None, optional
        The minimum value for normalization. If None, uses the minimum of the data.
    vmax : float | None, optional
        The maximum value for normalization. If None, uses the maximum of the data.
    show_hist : bool, optional
        Whether to display a histogram of the continuous values.
    hist_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the histogram.
    hist_axes_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to define the histogram Axes.

    Returns
    -------
    tuple[Axes, dict[Tree, Color]] | tuple[Axes, dict[Tree, Color], Axes]
        The Axes, a dictionary mapping each node to its assigned color,
        and optionally the histogram Axes if `show_hist` is True.
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)

    features = {node: node[color_by] for node in tree if color_by in node.metadata}
    values = list(features.values())
    vmin = min(values) if vmin is None else vmin
    vmax = max(values) if vmax is None else vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = {
        node: colormap(norm(float(features[node])))
        if node in features
        else default_color
        for node in tree
    }

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
        return ax, colors, hist_ax  # pyright: ignore

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    ax.get_figure().colorbar(sm, ax=ax)  # pyright: ignore
    return ax, colors


@overload
def draw_colored_tree_continuous(
    tree: Tree,
    color_by: str,
    ax: Axes | None = None,
    backward_time: bool = False,
    default_color: Color = "black",
    colormap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    branch_kwargs: dict[str, Any] | None = None,
    sampled_ancestor_kwargs: dict[str, Any] | None = None,
    *,
    show_hist: Literal[False],
    hist_kwargs: dict[str, Any] | None = None,
    hist_axes_kwargs: dict[str, Any] | None = None,
) -> Axes: ...
@overload
def draw_colored_tree_continuous(
    tree: Tree,
    color_by: str,
    ax: Axes | None = None,
    backward_time: bool = False,
    default_color: Color = "black",
    colormap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    branch_kwargs: dict[str, Any] | None = None,
    sampled_ancestor_kwargs: dict[str, Any] | None = None,
    show_hist: Literal[True] = True,
    hist_kwargs: dict[str, Any] | None = None,
    hist_axes_kwargs: dict[str, Any] | None = None,
) -> tuple[Axes, Axes]: ...
def draw_colored_tree_continuous(
    tree: Tree,
    color_by: str,
    ax: Axes | None = None,
    backward_time: bool = False,
    default_color: Color = "black",
    colormap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    branch_kwargs: dict[str, Any] | None = None,
    sampled_ancestor_kwargs: dict[str, Any] | None = None,
    show_hist: bool = True,
    hist_kwargs: dict[str, Any] | None = None,
    hist_axes_kwargs: dict[str, Any] | None = None,
) -> Axes | tuple[Axes, Axes]:
    """
    Draw a phylogenetic tree with branches colored based on continuous metadata.

    Parameters
    ----------
    tree : Tree
        The phylogenetic tree to draw.
    color_by : str
        The metadata key to color branches by.
    ax : Axes | None, optional
        The matplotlib Axes to draw on. If None, uses the current Axes.
    backward_time : bool, optional
        If True, the x-axis is inverted to represent time going backward.
    default_color : Color, optional
        The color to use for nodes without the specified metadata.
    colormap : str | Colormap, optional
        The colormap to use for coloring continuous values. Defaults to 'viridis'.
    vmin : float | None, optional
        The minimum value for normalization. If None, uses the minimum of the data.
    vmax : float | None, optional
        The maximum value for normalization. If None, uses the maximum of the data.
    branch_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the branch drawing functions.
    sampled_ancestor_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to highlight sampled ancestors.
    show_hist : bool, optional
        Whether to display a histogram of the continuous values.
    hist_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the histogram.
    hist_axes_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to define the histogram Axes.

    Returns
    -------
    Axes | tuple[Axes, Axes]
        The Axes with the drawn colored tree,
        and optionally the histogram Axes if `show_hist` is True.
    """
    if show_hist:
        ax, colors, hist_ax = _init_colored_tree_continuous(
            tree=tree,
            color_by=color_by,
            ax=ax,
            default_color=default_color,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            show_hist=show_hist,
            hist_kwargs=hist_kwargs,
            hist_axes_kwargs=hist_axes_kwargs,
        )
        return draw_tree(
            tree=tree,
            ax=ax,
            colors=colors,
            backward_time=backward_time,
            branch_kwargs=branch_kwargs,
            sampled_ancestor_kwargs=sampled_ancestor_kwargs,
        ), hist_ax

    ax, colors = _init_colored_tree_continuous(
        tree=tree,
        color_by=color_by,
        ax=ax,
        default_color=default_color,
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
        show_hist=show_hist,
        hist_kwargs=hist_kwargs,
        hist_axes_kwargs=hist_axes_kwargs,
    )
    return draw_tree(
        tree=tree,
        ax=ax,
        colors=colors,
        backward_time=backward_time,
        branch_kwargs=branch_kwargs,
        sampled_ancestor_kwargs=sampled_ancestor_kwargs,
    )


@overload
def draw_colored_dated_tree_continuous(
    tree: Tree,
    calibration_nodes: tuple[CalibrationNode, CalibrationNode],
    color_by: str,
    ax: Axes | None = None,
    default_color: Color = "black",
    colormap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    branch_kwargs: dict[str, Any] | None = None,
    *,
    show_hist: Literal[False],
    hist_kwargs: dict[str, Any] | None = None,
    hist_axes_kwargs: dict[str, Any] | None = None,
) -> Axes: ...
@overload
def draw_colored_dated_tree_continuous(
    tree: Tree,
    calibration_nodes: tuple[CalibrationNode, CalibrationNode],
    color_by: str,
    ax: Axes | None = None,
    default_color: Color = "black",
    colormap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    branch_kwargs: dict[str, Any] | None = None,
    show_hist: Literal[True] = True,
    hist_kwargs: dict[str, Any] | None = None,
    hist_axes_kwargs: dict[str, Any] | None = None,
) -> tuple[Axes, Axes]: ...
def draw_colored_dated_tree_continuous(
    tree: Tree,
    calibration_nodes: tuple[CalibrationNode, CalibrationNode],
    color_by: str,
    ax: Axes | None = None,
    default_color: Color = "black",
    colormap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    branch_kwargs: dict[str, Any] | None = None,
    show_hist: bool = True,
    hist_kwargs: dict[str, Any] | None = None,
    hist_axes_kwargs: dict[str, Any] | None = None,
) -> Axes | tuple[Axes, Axes]:
    """
    Draw a dated phylogenetic tree with branches colored based on continuous metadata.

    Parameters
    ----------
    tree : Tree
        The phylogenetic tree to draw.
    calibration_nodes : tuple[CalibrationNode, CalibrationNode]
        Two calibration nodes defining the mapping from depth to date.
    color_by : str
        The metadata key to color branches by.
    ax : Axes | None, optional
        The matplotlib Axes to draw on. If None, uses the current Axes.
    default_color : Color, optional
        The color to use for nodes without the specified metadata.
    colormap : str | Colormap, optional
        The colormap to use for coloring continuous values. Defaults to 'viridis'.
    vmin : float | None, optional
        The minimum value for normalization. If None, uses the minimum of the data.
    vmax : float | None, optional
        The maximum value for normalization. If None, uses the maximum of the data.
    branch_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the branch drawing functions.
    show_hist : bool, optional
        Whether to display a histogram of the continuous values.
    hist_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the histogram.
    hist_axes_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to define the histogram Axes.

    Returns
    -------
    Axes | tuple[Axes, Axes]
        The Axes with the drawn colored dated tree,
        and optionally the histogram Axes if `show_hist` is True.
    """
    if show_hist:
        ax, colors, hist_ax = _init_colored_tree_continuous(
            tree=tree,
            color_by=color_by,
            ax=ax,
            default_color=default_color,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            show_hist=show_hist,
            hist_kwargs=hist_kwargs,
            hist_axes_kwargs=hist_axes_kwargs,
        )
        return draw_dated_tree(
            tree=tree,
            calibration_nodes=calibration_nodes,
            ax=ax,
            colors=colors,
            branch_kwargs=branch_kwargs,
        ), hist_ax

    ax, colors = _init_colored_tree_continuous(
        tree=tree,
        color_by=color_by,
        ax=ax,
        default_color=default_color,
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
        show_hist=show_hist,
        hist_kwargs=hist_kwargs,
        hist_axes_kwargs=hist_axes_kwargs,
    )
    return draw_dated_tree(
        tree=tree,
        calibration_nodes=calibration_nodes,
        ax=ax,
        colors=colors,
        branch_kwargs=branch_kwargs,
    )
