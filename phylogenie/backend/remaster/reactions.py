from dataclasses import dataclass

import phylogenie.typings as pgt
from phylogenie.skyline import (
    SkylineMatrixLike,
    SkylineParameterLike,
    SkylineVectorLike,
    skyline_matrix,
    skyline_vector,
)

SAMPLE_POPULATION = "sample"
DEFAULT_POPULATION = "X"


@dataclass
class Reaction:
    rate: SkylineParameterLike
    value: str


@dataclass
class PunctualReaction:
    times: pgt.OneOrManyScalars
    value: str
    p: pgt.OneOrManyScalars | None = None
    n: pgt.OneOrMany[int] | None = None


def get_canonical_reactions(
    populations: str | list[str] = DEFAULT_POPULATION,
    sample_population: str = SAMPLE_POPULATION,
    birth_rates: SkylineVectorLike = 0,
    death_rates: SkylineVectorLike = 0,
    sampling_rates: SkylineVectorLike = 0,
    removal_probabilities: SkylineVectorLike = 0,
    migration_rates: SkylineMatrixLike = 0,
    birth_rates_among_demes: SkylineMatrixLike = 0,
) -> list[Reaction]:
    if isinstance(populations, str):
        populations = [populations]
    N = len(populations)

    birth_rates = skyline_vector(birth_rates, N=N)
    death_rates = skyline_vector(death_rates, N=N)
    sampling_rates = skyline_vector(sampling_rates, N=N)
    removal_probabilities = skyline_vector(removal_probabilities, N=N)
    migration_rates = skyline_matrix(migration_rates, N=N, zero_diagonal=True)
    birth_rates_among_demes = skyline_matrix(
        birth_rates_among_demes, N=N, zero_diagonal=True
    )

    reactions: list[Reaction] = []
    for i, population in enumerate(populations):
        reactions.append(Reaction(birth_rates[i], f"{population} -> 2{population}"))
        reactions.append(Reaction(death_rates[i], f"{population} -> 0"))
        reactions.append(
            Reaction(
                sampling_rates[i] * removal_probabilities[i],
                f"{population} -> {sample_population}",
            )
        )
        reactions.append(
            Reaction(
                sampling_rates[i] * (1 - removal_probabilities[i]),
                f"{population} -> {population} + {sample_population}",
            )
        )
        for j, other_population in enumerate(populations):
            if i == j:
                continue
            reactions.append(
                Reaction(migration_rates[i][j], f"{population} -> {other_population}")
            )
            reactions.append(
                Reaction(
                    birth_rates_among_demes[i][j],
                    f"{population} -> {population} + {other_population}",
                )
            )
    return reactions


def get_epidemiological_reactions(
    populations: str | list[str] = DEFAULT_POPULATION,
    sample_population: str = SAMPLE_POPULATION,
    reproduction_numbers: SkylineVectorLike = 0,
    become_uninfectious_rates: SkylineVectorLike = 0,
    sampling_proportions: SkylineVectorLike = 0,
    removal_probabilities: SkylineVectorLike = 0,
    migration_rates: SkylineMatrixLike = 0,
    reproduction_numbers_among_demes: SkylineMatrixLike = 0,
) -> list[Reaction]:
    if isinstance(populations, str):
        populations = [populations]
    N = len(populations)

    reproduction_numbers = skyline_vector(reproduction_numbers, N=N)
    become_uninfectious_rates = skyline_vector(become_uninfectious_rates, N=N)
    sampling_proportions = skyline_vector(sampling_proportions, N=N)
    removal_probabilities = skyline_vector(removal_probabilities, N=N)
    reproduction_numbers_among_demes = skyline_matrix(
        reproduction_numbers_among_demes, N=N, zero_diagonal=True
    )

    birth_rates = reproduction_numbers * become_uninfectious_rates
    birth_rates_among_demes = (
        reproduction_numbers_among_demes * become_uninfectious_rates
    )
    sampling_rates = become_uninfectious_rates * sampling_proportions
    death_rates = become_uninfectious_rates - removal_probabilities * sampling_rates

    return get_canonical_reactions(
        populations=populations,
        sample_population=sample_population,
        birth_rates=birth_rates,
        death_rates=death_rates,
        sampling_rates=sampling_rates,
        removal_probabilities=removal_probabilities,
        migration_rates=migration_rates,
        birth_rates_among_demes=birth_rates_among_demes,
    )


def get_FBD_reactions(
    populations: str | list[str] = DEFAULT_POPULATION,
    sample_population: str = SAMPLE_POPULATION,
    diversification: SkylineVectorLike = 0,
    turnover: SkylineVectorLike = 0,
    sampling_proportions: SkylineVectorLike = 0,
    removal_probabilities: SkylineVectorLike = 0,
    migration_rates: SkylineMatrixLike = 0,
    diversification_between_types: SkylineMatrixLike = 0,
):
    if isinstance(populations, str):
        populations = [populations]
    N = len(populations)

    diversification = skyline_vector(diversification, N=N)
    turnover = skyline_vector(turnover, N=N)
    sampling_proportions = skyline_vector(sampling_proportions, N=N)
    removal_probabilities = skyline_vector(removal_probabilities, N=N)
    diversification_between_types = skyline_matrix(
        diversification_between_types, N=N, zero_diagonal=True
    )

    birth_rates = diversification / (1 - turnover)
    death_rates = turnover * birth_rates
    sampling_rates = (
        sampling_proportions
        * death_rates
        / (1 - removal_probabilities * sampling_proportions)
    )
    birth_rates_among_demes = diversification_between_types + death_rates

    return get_canonical_reactions(
        populations=populations,
        sample_population=sample_population,
        birth_rates=birth_rates,
        death_rates=death_rates,
        sampling_rates=sampling_rates,
        removal_probabilities=removal_probabilities,
        migration_rates=migration_rates,
        birth_rates_among_demes=birth_rates_among_demes,
    )
