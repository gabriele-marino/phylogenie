from collections.abc import Sequence

from phylogenie.skyline import (
    SkylineMatrixCoercible,
    SkylineParameterLike,
    SkylineVectorCoercible,
    skyline_matrix,
    skyline_vector,
)
from phylogenie.treesimulator.unbounded_population.model import UnboundedPopulationModel

INFECTIOUS_STATE = "I"
EXPOSED_STATE = "E"
SUPERSPREADER_STATE = "S"


def get_canonical_model(
    init_state: str,
    states: Sequence[str],
    sampling_rates: SkylineVectorCoercible = 0,
    remove_after_sampling: bool = False,
    birth_rates: SkylineVectorCoercible = 0,
    death_rates: SkylineVectorCoercible = 0,
    migration_rates: SkylineMatrixCoercible | None = None,
    birth_rates_among_states: SkylineMatrixCoercible | None = None,
    max_time: float | None = None,
    seed: int | None = None,
) -> UnboundedPopulationModel:
    N = len(states)

    birth_rates = skyline_vector(birth_rates, N)
    death_rates = skyline_vector(death_rates, N)
    sampling_rates = skyline_vector(sampling_rates, N)

    model = UnboundedPopulationModel(
        init_state=init_state, max_time=max_time, seed=seed
    )
    for i, state in enumerate(states):
        model.add_birth_event(state, state, birth_rates[i])
        model.add_death_event(state, death_rates[i])
        model.add_sampling_event(state, remove_after_sampling, sampling_rates[i])

    if migration_rates is not None:
        migration_rates = skyline_matrix(migration_rates, N, N - 1)
        for i, state in enumerate(states):
            for j, other_state in enumerate([s for s in states if s != state]):
                model.add_migration_event(state, other_state, migration_rates[i, j])

    if birth_rates_among_states is not None:
        birth_rates_among_states = skyline_matrix(birth_rates_among_states, N, N - 1)
        for i, state in enumerate(states):
            for j, other_state in enumerate([s for s in states if s != state]):
                model.add_birth_event(
                    state, other_state, birth_rates_among_states[i, j]
                )

    return model


def get_FBD_model(
    init_state: str,
    states: Sequence[str],
    sampling_proportions: SkylineVectorCoercible = 0,
    diversification: SkylineVectorCoercible = 0,
    turnover: SkylineVectorCoercible = 0,
    migration_rates: SkylineMatrixCoercible | None = None,
    diversification_between_states: SkylineMatrixCoercible | None = None,
    max_time: float | None = None,
    seed: int | None = None,
) -> UnboundedPopulationModel:
    N = len(states)

    diversification = skyline_vector(diversification, N)
    turnover = skyline_vector(turnover, N)
    sampling_proportions = skyline_vector(sampling_proportions, N)

    birth_rates = diversification / (1 - turnover)
    death_rates = turnover * birth_rates
    sampling_rates = sampling_proportions * death_rates
    birth_rates_among_states = (
        (skyline_matrix(diversification_between_states, N, N - 1) + death_rates)
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
        max_time=max_time,
        seed=seed,
    )


def get_epidemiological_model(
    init_state: str,
    states: Sequence[str],
    sampling_proportions: SkylineVectorCoercible,
    reproduction_numbers: SkylineVectorCoercible = 0,
    become_uninfectious_rates: SkylineVectorCoercible = 0,
    migration_rates: SkylineMatrixCoercible | None = None,
    reproduction_numbers_among_states: SkylineMatrixCoercible | None = None,
    max_time: float | None = None,
    seed: int | None = None,
) -> UnboundedPopulationModel:
    N = len(states)

    reproduction_numbers = skyline_vector(reproduction_numbers, N)
    become_uninfectious_rates = skyline_vector(become_uninfectious_rates, N)
    sampling_proportions = skyline_vector(sampling_proportions, N)

    birth_rates = reproduction_numbers * become_uninfectious_rates
    sampling_rates = become_uninfectious_rates * sampling_proportions
    death_rates = become_uninfectious_rates - sampling_rates
    birth_rates_among_states = (
        (
            skyline_matrix(reproduction_numbers_among_states, N, N - 1)
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
        max_time=max_time,
        seed=seed,
    )


def get_BD_model(
    reproduction_number: SkylineParameterLike,
    infectious_period: SkylineParameterLike,
    sampling_proportion: SkylineParameterLike,
    max_time: float | None = None,
    seed: int | None = None,
) -> UnboundedPopulationModel:
    return get_epidemiological_model(
        init_state=INFECTIOUS_STATE,
        states=[INFECTIOUS_STATE],
        reproduction_numbers=reproduction_number,
        become_uninfectious_rates=1 / infectious_period,
        sampling_proportions=sampling_proportion,
        max_time=max_time,
        seed=seed,
    )


def get_BDEI_model(
    init_state: str,
    reproduction_number: SkylineParameterLike,
    infectious_period: SkylineParameterLike,
    incubation_period: SkylineParameterLike,
    sampling_proportion: SkylineParameterLike,
    max_time: float | None = None,
    seed: int | None = None,
) -> UnboundedPopulationModel:
    return get_epidemiological_model(
        init_state=init_state,
        states=[EXPOSED_STATE, INFECTIOUS_STATE],
        sampling_proportions=[0, sampling_proportion],
        become_uninfectious_rates=[0, 1 / infectious_period],
        reproduction_numbers_among_states=[[0], [reproduction_number]],
        migration_rates=[[1 / incubation_period], [0]],
        max_time=max_time,
        seed=seed,
    )


def get_BDSS_model(
    init_state: str,
    reproduction_number: SkylineParameterLike,
    infectious_period: SkylineParameterLike,
    superspreading_ratio: SkylineParameterLike,
    superspreaders_proportion: SkylineParameterLike,
    sampling_proportion: SkylineParameterLike,
    max_time: float | None = None,
    seed: int | None = None,
) -> UnboundedPopulationModel:
    f_SS = superspreaders_proportion
    r_SS = superspreading_ratio
    R_0_IS = reproduction_number * f_SS / (1 + r_SS * f_SS - f_SS)
    R_0_SI = (reproduction_number - r_SS * R_0_IS) * r_SS
    R_0_S = r_SS * R_0_IS
    R_0_I = R_0_SI / r_SS
    return get_epidemiological_model(
        init_state=init_state,
        states=[INFECTIOUS_STATE, SUPERSPREADER_STATE],
        reproduction_numbers=[R_0_I, R_0_S],
        reproduction_numbers_among_states=[[R_0_IS], [R_0_SI]],
        become_uninfectious_rates=1 / infectious_period,
        sampling_proportions=sampling_proportion,
        max_time=max_time,
        seed=seed,
    )
