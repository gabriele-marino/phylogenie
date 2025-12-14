from collections.abc import Iterable
from enum import Enum

from phylogenie.treesimulator.events.mutations import get_mutation_id
from phylogenie.treesimulator.model import get_node_state
from phylogenie.treesimulator.tree import Tree
from phylogenie.treesimulator.utils import (
    get_node_depth_levels,
    get_node_depths,
    get_node_height_levels,
    get_node_heights,
    get_node_leaf_counts,
)


def _get_states(tree: Tree) -> dict[Tree, str]:
    return {node: get_node_state(node.name) for node in tree}


def _get_mutations(tree: Tree) -> dict[Tree, int]:
    return {node: get_mutation_id(node.name) for node in tree}


class Feature(str, Enum):
    DEPTH = "depth"
    DEPTH_LEVEL = "depth_level"
    HEIGHT = "height"
    HEIGHT_LEVEL = "height_level"
    MUTATION = "mutation"
    N_LEAVES = "n_leaves"
    STATE = "state"


FEATURES_EXTRACTORS = {
    Feature.DEPTH: get_node_depths,
    Feature.DEPTH_LEVEL: get_node_depth_levels,
    Feature.HEIGHT: get_node_heights,
    Feature.HEIGHT_LEVEL: get_node_height_levels,
    Feature.MUTATION: _get_mutations,
    Feature.N_LEAVES: get_node_leaf_counts,
    Feature.STATE: _get_states,
}


def set_features(tree: Tree, features: Iterable[Feature]) -> None:
    for feature in features:
        feature_maps = FEATURES_EXTRACTORS[feature](tree)
        for node in tree:
            node[feature.value] = feature_maps[node]
