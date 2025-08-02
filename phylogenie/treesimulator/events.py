from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phylogenie.skyline import (
    SkylineMatrixCoercible,
    SkylineParameter,
    SkylineParameterLike,
    SkylineVectorCoercible,
    skyline_matrix,
    skyline_parameter,
    skyline_vector,
)
from phylogenie.treesimulator.model import Model, get_CT_state

INFECTIOUS_STATE = "I"
EXPOSED_STATE = "E"
SUPERSPREADER_STATE = "S"


@dataclass
class Event(ABC):
    rate: SkylineParameter
    state: str

    def get_propensity(self, model: Model, time: float) -> float:
        n_individuals = model.count_individuals(self.state)
        rate = self.rate.get_value_at_time(time)
        if rate == np.inf and not n_individuals:
            return 0
        return rate * n_individuals

    @abstractmethod
    def apply(self, model: Model, time: float) -> None: ...


@dataclass
class BirthEvent(Event):
    child_state: str

    def apply(self, model: Model, time: float) -> None:
        individual = model.get_random_individual(self.state)
        model.birth_from(individual, self.child_state, time)


class DeathEvent(Event):
    def apply(self, model: Model, time: float) -> None:
        individual = model.get_random_individual(self.state)
        model.remove(individual, time)


@dataclass
class MigrationEvent(Event):
    target_state: str

    def apply(self, model: Model, time: float) -> None:
        individual = model.get_random_individual(self.state)
        model.migrate(individual, self.target_state, time)


@dataclass
class SamplingEvent(Event):
    removal_probability: SkylineParameter

    def apply(self, model: Model, time: float) -> None:
        individual = model.get_random_individual(self.state)
        model.sample(individual, time, self.removal_probability.get_value_at_time(time))


def get_canonical_events(
    states: Sequence[str],
    sampling_rates: SkylineVectorCoercible,
    birth_rates: SkylineVectorCoercible = 0,
    death_rates: SkylineVectorCoercible = 0,
    removal_probabilities: SkylineVectorCoercible = 0,
    migration_rates: SkylineMatrixCoercible | None = None,
    birth_rates_among_states: SkylineMatrixCoercible | None = None,
) -> list[Event]:
    N = len(states)

    birth_rates = skyline_vector(birth_rates, N)
    death_rates = skyline_vector(death_rates, N)
    sampling_rates = skyline_vector(sampling_rates, N)
    removal_probabilities = skyline_vector(removal_probabilities, N)

    events: list[Event] = []
    for i, state in enumerate(states):
        events.append(BirthEvent(birth_rates[i], state, state))
        events.append(DeathEvent(death_rates[i], state))
        events.append(SamplingEvent(sampling_rates[i], state, removal_probabilities[i]))

    if migration_rates is not None:
        migration_rates = skyline_matrix(migration_rates, N, N - 1)
        for i, state in enumerate(states):
            for j, other_state in enumerate([s for s in states if s != state]):
                events.append(MigrationEvent(migration_rates[i, j], state, other_state))

    if birth_rates_among_states is not None:
        birth_rates_among_states = skyline_matrix(birth_rates_among_states, N, N - 1)
        for i, state in enumerate(states):
            for j, other_state in enumerate([s for s in states if s != state]):
                events.append(
                    BirthEvent(birth_rates_among_states[i, j], state, other_state)
                )

    return [event for event in events if event.rate]


def get_epidemiological_events(
    states: Sequence[str],
    sampling_proportions: SkylineVectorCoercible = 1,
    reproduction_numbers: SkylineVectorCoercible = 0,
    become_uninfectious_rates: SkylineVectorCoercible = 0,
    removal_probabilities: SkylineVectorCoercible = 1,
    migration_rates: SkylineMatrixCoercible | None = None,
    reproduction_numbers_among_states: SkylineMatrixCoercible | None = None,
) -> list[Event]:
    N = len(states)

    reproduction_numbers = skyline_vector(reproduction_numbers, N)
    become_uninfectious_rates = skyline_vector(become_uninfectious_rates, N)
    sampling_proportions = skyline_vector(sampling_proportions, N)
    removal_probabilities = skyline_vector(removal_probabilities, N)

    birth_rates = reproduction_numbers * become_uninfectious_rates
    sampling_rates = become_uninfectious_rates * sampling_proportions
    death_rates = become_uninfectious_rates - removal_probabilities * sampling_rates
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
        birth_rates=birth_rates,
        death_rates=death_rates,
        sampling_rates=sampling_rates,
        removal_probabilities=removal_probabilities,
        migration_rates=migration_rates,
        birth_rates_among_states=birth_rates_among_states,
    )


def get_FBD_events(
    states: Sequence[str],
    diversification: SkylineVectorCoercible = 0,
    turnover: SkylineVectorCoercible = 0,
    sampling_proportions: SkylineVectorCoercible = 1,
    removal_probabilities: SkylineVectorCoercible = 0,
    migration_rates: SkylineMatrixCoercible | None = None,
    diversification_between_types: SkylineMatrixCoercible | None = None,
) -> list[Event]:
    N = len(states)

    diversification = skyline_vector(diversification, N)
    turnover = skyline_vector(turnover, N)
    sampling_proportions = skyline_vector(sampling_proportions, N)
    removal_probabilities = skyline_vector(removal_probabilities, N)

    birth_rates = diversification / (1 - turnover)
    death_rates = turnover * birth_rates
    sampling_rates_dividend = 1 - removal_probabilities * sampling_proportions
    sampling_rates = sampling_proportions * death_rates / sampling_rates_dividend
    birth_rates_among_states = (
        (skyline_matrix(diversification_between_types, N, N - 1) + death_rates)
        if diversification_between_types is not None
        else None
    )

    return get_canonical_events(
        states=states,
        birth_rates=birth_rates,
        death_rates=death_rates,
        sampling_rates=sampling_rates,
        removal_probabilities=removal_probabilities,
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


def get_contact_tracing_events(
    events: Sequence[Event],
    samplable_states_after_notification: Sequence[str] | None = None,
    sampling_rate_after_notification: SkylineParameterLike = np.inf,
    contacts_removal_probability: SkylineParameterLike = 1,
) -> list[Event]:
    ct_events = list(events)
    for event in events:
        if isinstance(event, MigrationEvent):
            ct_events.append(
                MigrationEvent(
                    event.rate,
                    get_CT_state(event.state),
                    get_CT_state(event.target_state),
                )
            )
        elif isinstance(event, BirthEvent):
            ct_events.append(
                BirthEvent(event.rate, get_CT_state(event.state), event.child_state)
            )

    for state in (
        samplable_states_after_notification
        if samplable_states_after_notification is not None
        else [e.state for e in events]
    ):
        ct_events.append(
            SamplingEvent(
                skyline_parameter(sampling_rate_after_notification),
                get_CT_state(state),
                skyline_parameter(contacts_removal_probability),
            )
        )

    return ct_events
