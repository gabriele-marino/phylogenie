from collections.abc import Sequence
from dataclasses import dataclass

from phylogenie.skyline import (
    SkylineMatrixCoercible,
    SkylineParameterLike,
    SkylineVectorCoercible,
    skyline_matrix,
    skyline_vector,
)
from phylogenie.tree_node import TreeNode
from phylogenie.treesimulator.events import (
    Death,
    Migration,
    Sampling,
    SingleReactantEventFunction,
    StochasticEvent,
)
from phylogenie.treesimulator.model import Model

INFECTIOUS_STATE = "I"
EXPOSED_STATE = "E"
SUPERSPREADER_STATE = "S"


@dataclass(kw_only=True)
class Birth(SingleReactantEventFunction):
    new_state: str

    def apply_to_node(self, model: Model, node: TreeNode):
        model.birth_from(node, self.new_state)


def get_canonical_model(
    init_state: str,
    states: Sequence[str],
    sampling_rates: SkylineVectorCoercible = 0,
    remove_after_sampling: bool = False,
    birth_rates: SkylineVectorCoercible = 0,
    death_rates: SkylineVectorCoercible = 0,
    migration_rates: SkylineMatrixCoercible | None = None,
    birth_rates_among_states: SkylineMatrixCoercible | None = None,
) -> Model:
    n_states = len(states)

    birth_rates = skyline_vector(birth_rates, n_states)
    death_rates = skyline_vector(death_rates, n_states)
    sampling_rates = skyline_vector(sampling_rates, n_states)

    model = Model(init_state=init_state)
    for i, state in enumerate(states):
        model.add_event(
            StochasticEvent(rate=birth_rates[i], fn=Birth(state=state, new_state=state))
        )
        model.add_event(StochasticEvent(rate=death_rates[i], fn=Death(state=state)))
        model.add_event(
            StochasticEvent(
                rate=sampling_rates[i],
                fn=Sampling(state=state, removal=remove_after_sampling),
            )
        )

    if migration_rates is not None:
        migration_rates = skyline_matrix(migration_rates, n_states, n_states - 1)
        for i, state in enumerate(states):
            for j, other_state in enumerate([s for s in states if s != state]):
                model.add_event(
                    StochasticEvent(
                        rate=migration_rates[i, j],
                        fn=Migration(state=state, target_state=other_state),
                    )
                )

    if birth_rates_among_states is not None:
        birth_rates_among_states = skyline_matrix(
            birth_rates_among_states, n_states, n_states - 1
        )
        for i, state in enumerate(states):
            for j, other_state in enumerate([s for s in states if s != state]):
                model.add_event(
                    StochasticEvent(
                        rate=birth_rates_among_states[i, j],
                        fn=Birth(state=state, new_state=other_state),
                    )
                )

    return model


def get_fbd_model(
    init_state: str,
    states: Sequence[str],
    sampling_proportions: SkylineVectorCoercible = 0,
    diversification: SkylineVectorCoercible = 0,
    turnover: SkylineVectorCoercible = 0,
    migration_rates: SkylineMatrixCoercible | None = None,
    diversification_between_states: SkylineMatrixCoercible | None = None,
) -> Model:
    n_states = len(states)

    diversification = skyline_vector(diversification, n_states)
    turnover = skyline_vector(turnover, n_states)
    sampling_proportions = skyline_vector(sampling_proportions, n_states)

    birth_rates = diversification / (1 - turnover)
    death_rates = turnover * birth_rates
    sampling_rates = sampling_proportions * death_rates
    birth_rates_among_states = (
        (
            skyline_matrix(diversification_between_states, n_states, n_states - 1)
            + death_rates
        )
        if diversification_between_states is not None
        else None
    )

    return get_canonical_model(
        init_state=init_state,
        states=states,
        sampling_rates=sampling_rates,
        remove_after_sampling=False,
        birth_rates=birth_rates,
        death_rates=death_rates,
        migration_rates=migration_rates,
        birth_rates_among_states=birth_rates_among_states,
    )


def get_epidemiological_model(
    init_state: str,
    states: Sequence[str],
    sampling_proportions: SkylineVectorCoercible,
    reproduction_numbers: SkylineVectorCoercible = 0,
    become_uninfectious_rates: SkylineVectorCoercible = 0,
    migration_rates: SkylineMatrixCoercible | None = None,
    reproduction_numbers_among_states: SkylineMatrixCoercible | None = None,
) -> Model:
    n_states = len(states)

    reproduction_numbers = skyline_vector(reproduction_numbers, n_states)
    become_uninfectious_rates = skyline_vector(become_uninfectious_rates, n_states)
    sampling_proportions = skyline_vector(sampling_proportions, n_states)

    birth_rates = reproduction_numbers * become_uninfectious_rates
    sampling_rates = become_uninfectious_rates * sampling_proportions
    death_rates = become_uninfectious_rates - sampling_rates
    birth_rates_among_states = (
        (
            skyline_matrix(reproduction_numbers_among_states, n_states, n_states - 1)
            * become_uninfectious_rates
        )
        if reproduction_numbers_among_states is not None
        else None
    )

    return get_canonical_model(
        init_state=init_state,
        states=states,
        sampling_rates=sampling_rates,
        remove_after_sampling=True,
        birth_rates=birth_rates,
        death_rates=death_rates,
        migration_rates=migration_rates,
        birth_rates_among_states=birth_rates_among_states,
    )


def get_bd_model(
    reproduction_number: SkylineParameterLike,
    infectious_period: SkylineParameterLike,
    sampling_proportion: SkylineParameterLike,
) -> Model:
    return get_epidemiological_model(
        init_state=INFECTIOUS_STATE,
        states=[INFECTIOUS_STATE],
        reproduction_numbers=reproduction_number,
        become_uninfectious_rates=1 / infectious_period,
        sampling_proportions=sampling_proportion,
    )


def get_bdei_model(
    init_state: str,
    reproduction_number: SkylineParameterLike,
    infectious_period: SkylineParameterLike,
    incubation_period: SkylineParameterLike,
    sampling_proportion: SkylineParameterLike,
) -> Model:
    return get_epidemiological_model(
        init_state=init_state,
        states=[EXPOSED_STATE, INFECTIOUS_STATE],
        sampling_proportions=[0, sampling_proportion],
        become_uninfectious_rates=[0, 1 / infectious_period],
        reproduction_numbers_among_states=[[0], [reproduction_number]],
        migration_rates=[[1 / incubation_period], [0]],
    )


def get_bdss_model(
    init_state: str,
    reproduction_number: SkylineParameterLike,
    infectious_period: SkylineParameterLike,
    superspreading_ratio: SkylineParameterLike,
    superspreaders_proportion: SkylineParameterLike,
    sampling_proportion: SkylineParameterLike,
) -> Model:
    f_ss = superspreaders_proportion
    r_ss = superspreading_ratio
    r0_is = reproduction_number * f_ss / (1 + r_ss * f_ss - f_ss)
    r0_si = (reproduction_number - r_ss * r0_is) * r_ss
    r0_s = r_ss * r0_is
    r0_i = r0_si / r_ss
    return get_epidemiological_model(
        init_state=init_state,
        states=[INFECTIOUS_STATE, SUPERSPREADER_STATE],
        reproduction_numbers=[r0_i, r0_s],
        reproduction_numbers_among_states=[[r0_is], [r0_si]],
        become_uninfectious_rates=1 / infectious_period,
        sampling_proportions=sampling_proportion,
    )
