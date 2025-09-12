from collections.abc import Iterable
from enum import Enum

from phylogenie.tree import Tree
from phylogenie.treesimulator.events import get_mutation_id
from phylogenie.treesimulator.model import get_node_state
from phylogenie.utils import get_heights, get_n_tips, get_times


def _get_states(tree: Tree) -> dict[str, str]:
    return {node.name: get_node_state(node.name) for node in tree}


def _get_mutations(tree: Tree) -> dict[str, int]:
    return {node.name: get_mutation_id(node.name) for node in tree}


class Feature(str, Enum):
    STATE = "state"
    MUTATION = "mutation"
    N_TIPS = "n_tips"
    TIME = "time"
    HEIGHT = "height"


FEATURES_EXTRACTORS = {
    Feature.STATE: _get_states,
    Feature.MUTATION: _get_mutations,
    Feature.N_TIPS: get_n_tips,
    Feature.TIME: get_times,
    Feature.HEIGHT: get_heights,
}


def set_features(tree: Tree, features: Iterable[Feature]) -> None:
    for feature in features:
        feature_maps = FEATURES_EXTRACTORS[feature](tree)
        for node in tree:
            node.set(feature.value, feature_maps[node.name])
