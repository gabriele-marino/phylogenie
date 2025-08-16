from numpy.random import Generator

from phylogenie.skyline import (
    SkylineMatrixCoercible,
    SkylineParameterLike,
    SkylineVectorCoercible,
    skyline_matrix,
    skyline_vector,
)
from phylogenie.treesimulator.model import Event, Model

INFECTIOUS_STATE = "I"
EXPOSED_STATE = "E"
SUPERSPREADER_STATE = "S"


class Birth(Event):
    def __init__(self, state: str, rate: SkylineParameterLike, child_state: str):
        super().__init__(state, rate)
        self.child_state = child_state

    def apply(self, model: Model, time: float, rng: Generator) -> None:
        individual = self.draw_individual(model, rng)
        model.birth_from(individual, self.child_state, time)

    def __repr__(self) -> str:
        return f"Birth(state={self.state}, rate={self.rate}, child_state={self.child_state})"


class Death(Event):
    def apply(self, model: Model, time: float, rng: Generator) -> None:
        individual = self.draw_individual(model, rng)
        model.remove(individual, time)

    def __repr__(self) -> str:
        return f"Death(state={self.state}, rate={self.rate})"


class Migration(Event):
    def __init__(self, state: str, rate: SkylineParameterLike, target_state: str):
        super().__init__(state, rate)
        self.target_state = target_state

    def apply(self, model: Model, time: float, rng: Generator) -> None:
        individual = self.draw_individual(model, rng)
        model.migrate(individual, self.target_state, time)

    def __repr__(self) -> str:
        return f"Migration(state={self.state}, rate={self.rate}, target_state={self.target_state})"


class Sampling(Event):
    def __init__(self, state: str, rate: SkylineParameterLike, removal: bool):
        super().__init__(state, rate)
        self.removal = removal

    def apply(self, model: Model, time: float, rng: Generator) -> None:
        individual = self.draw_individual(model, rng)
        model.sample(individual, time, self.removal)

    def __repr__(self) -> str:
        return f"Sampling(state={self.state}, rate={self.rate}, removal={self.removal})"


def get_canonical_events(
    states: list[str],
    sampling_rates: SkylineVectorCoercible,
    remove_after_sampling: bool,
    birth_rates: SkylineVectorCoercible = 0,
    death_rates: SkylineVectorCoercible = 0,
    migration_rates: SkylineMatrixCoercible | None = None,
    birth_rates_among_states: SkylineMatrixCoercible | None = None,
) -> list[Event]:
    N = len(states)

    birth_rates = skyline_vector(birth_rates, N)
    death_rates = skyline_vector(death_rates, N)
    sampling_rates = skyline_vector(sampling_rates, N)

    events: list[Event] = []
    for i, state in enumerate(states):
        events.append(Birth(state, birth_rates[i], state))
        events.append(Death(state, death_rates[i]))
        events.append(Sampling(state, sampling_rates[i], remove_after_sampling))

    if migration_rates is not None:
        migration_rates = skyline_matrix(migration_rates, N, N - 1)
        for i, state in enumerate(states):
            for j, other_state in enumerate([s for s in states if s != state]):
                events.append(Migration(state, migration_rates[i, j], other_state))

    if birth_rates_among_states is not None:
        birth_rates_among_states = skyline_matrix(birth_rates_among_states, N, N - 1)
        for i, state in enumerate(states):
            for j, other_state in enumerate([s for s in states if s != state]):
                events.append(Birth(state, birth_rates_among_states[i, j], other_state))

    return [event for event in events if event.rate]


def get_FBD_events(
    states: list[str],
    sampling_proportions: SkylineVectorCoercible = 1,
    diversification: SkylineVectorCoercible = 0,
    turnover: SkylineVectorCoercible = 0,
    migration_rates: SkylineMatrixCoercible | None = None,
    diversification_between_states: SkylineMatrixCoercible | None = None,
) -> list[Event]:
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

    return get_canonical_events(
        states=states,
        sampling_rates=sampling_rates,
        remove_after_sampling=False,
        birth_rates=birth_rates,
        death_rates=death_rates,
        migration_rates=migration_rates,
        birth_rates_among_states=birth_rates_among_states,
    )


def get_epidemiological_events(
    states: list[str],
    sampling_proportions: SkylineVectorCoercible = 1,
    reproduction_numbers: SkylineVectorCoercible = 0,
    become_uninfectious_rates: SkylineVectorCoercible = 0,
    migration_rates: SkylineMatrixCoercible | None = None,
    reproduction_numbers_among_states: SkylineMatrixCoercible | None = None,
) -> list[Event]:
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

    return get_canonical_events(
        states=states,
        sampling_rates=sampling_rates,
        remove_after_sampling=True,
        birth_rates=birth_rates,
        death_rates=death_rates,
        migration_rates=migration_rates,
        birth_rates_among_states=birth_rates_among_states,
    )


def get_BD_events(
    reproduction_number: SkylineParameterLike,
    infectious_period: SkylineParameterLike,
    sampling_proportion: SkylineParameterLike = 1,
) -> list[Event]:
    return get_epidemiological_events(
        states=[INFECTIOUS_STATE],
        reproduction_numbers=reproduction_number,
        become_uninfectious_rates=1 / infectious_period,
        sampling_proportions=sampling_proportion,
    )


def get_BDEI_events(
    reproduction_number: SkylineParameterLike,
    infectious_period: SkylineParameterLike,
    incubation_period: SkylineParameterLike,
    sampling_proportion: SkylineParameterLike = 1,
) -> list[Event]:
    return get_epidemiological_events(
        states=[EXPOSED_STATE, INFECTIOUS_STATE],
        sampling_proportions=[0, sampling_proportion],
        become_uninfectious_rates=[0, 1 / infectious_period],
        reproduction_numbers_among_states=[[0], [reproduction_number]],
        migration_rates=[[1 / incubation_period], [0]],
    )


def get_BDSS_events(
    reproduction_number: SkylineParameterLike,
    infectious_period: SkylineParameterLike,
    superspreading_ratio: SkylineParameterLike,
    superspreaders_proportion: SkylineParameterLike,
    sampling_proportion: SkylineParameterLike = 1,
) -> list[Event]:
    f_SS = superspreaders_proportion
    r_SS = superspreading_ratio
    R_0_IS = reproduction_number * f_SS / (1 + r_SS * f_SS - f_SS)
    R_0_SI = (reproduction_number - r_SS * R_0_IS) * r_SS
    R_0_S = r_SS * R_0_IS
    R_0_I = R_0_SI / r_SS
    return get_epidemiological_events(
        states=[INFECTIOUS_STATE, SUPERSPREADER_STATE],
        reproduction_numbers=[R_0_I, R_0_S],
        reproduction_numbers_among_states=[[R_0_IS], [R_0_SI]],
        become_uninfectious_rates=1 / infectious_period,
        sampling_proportions=sampling_proportion,
    )
