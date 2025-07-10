import phylogenie.core.trees.remaster.configs as cfg
import phylogenie.typings as pgt
from phylogenie.backend.remaster import PunctualReaction, Reaction
from phylogenie.core.factories import (
    many_ints_factory,
    many_scalars_factory,
    skyline_parameter_like_factory,
)


def reaction_factory(x: cfg.ReactionConfig, data: pgt.Data) -> Reaction:
    return Reaction(
        rate=skyline_parameter_like_factory(x.rate, data),
        value=x.value,
    )


def punctual_reaction_factory(
    x: cfg.PunctualReactionConfig, data: pgt.Data
) -> PunctualReaction:
    return PunctualReaction(
        times=many_scalars_factory(x.times, data),
        value=x.value,
        p=None if x.p is None else many_scalars_factory(x.p, data),
        n=None if x.n is None else many_ints_factory(x.n, data),
    )
