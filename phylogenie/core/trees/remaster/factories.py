import phylogenie.core.trees.remaster.configs as cfg
from phylogenie.backend.remaster import PunctualReaction, Reaction
from phylogenie.core.factories import (
    one_or_many_ints_factory,
    one_or_many_scalars_factory,
    skyline_parameter_like_factory,
)
from phylogenie.core.typings import Data


def reaction_factory(
    x: cfg.ReactionConfig,
    data: Data,
) -> Reaction:
    return Reaction(
        rate=skyline_parameter_like_factory(x.rate, data),
        value=x.value,
    )


def punctual_reaction_factory(
    x: cfg.PunctualReactionConfig,
    data: Data,
) -> PunctualReaction:
    return PunctualReaction(
        times=one_or_many_scalars_factory(x.times, data),
        value=x.value,
        p=None if x.p is None else one_or_many_scalars_factory(x.p, data),
        n=None if x.n is None else one_or_many_ints_factory(x.n, data),
    )
