from dataclasses import dataclass

import numpy as np
from treesimulator import STATE, save_forest
from treesimulator.generator import generate
from treesimulator.mtbd_models import CTModel, Model

from phylogenie.skyline import (
    SkylineMatrixLike,
    SkylineParameterLike,
    SkylineVectorLike,
    skyline_matrix,
    skyline_parameter,
    skyline_vector,
)

DEFAULT_POPULATION = "X"
INFECTIOUS_POPULATION = "I"
EXPOSED_POPULATION = "E"


@dataclass
class TreeParams:
    populations: str | list[str] = DEFAULT_POPULATION
    transition_rates: SkylineMatrixLike = 0
    transmission_rates: SkylineMatrixLike = 0
    removal_rates: SkylineVectorLike = 0
    sampling_proportions: SkylineVectorLike = 0


def generate_tree(
    output_file: str,
    params: TreeParams,
    min_tips: int,
    max_tips: int,
    T: float = np.inf,
    state_frequencies: list[float] | None = None,
    notification_probability: SkylineParameterLike = 0,
    notification_sampling_rate: SkylineParameterLike = np.inf,
    allow_irremovable_states: bool = False,
    max_notified_contacts: int = 1,
    root_state: str | None = None,
    random_seed: int | None = None,
) -> None:
    populations = params.populations
    if isinstance(populations, str):
        populations = [populations]
    N = len(populations)

    transition_rates = skyline_matrix(params.transition_rates, N=N, zero_diagonal=True)
    transmission_rates = skyline_matrix(params.transmission_rates, N=N)
    removal_rates = skyline_vector(params.removal_rates, N=N)
    sampling_proportions = skyline_vector(params.sampling_proportions, N=N)

    change_times = sorted(
        set(
            [
                *transition_rates.change_times,
                *transmission_rates.change_times,
                *removal_rates.change_times,
                *sampling_proportions.change_times,
            ]
        )
    )

    models = [
        Model(
            states=populations,
            transition_rates=transition_rates.get_value_at_time(t),
            transmission_rates=transmission_rates.get_value_at_time(t),
            removal_rates=removal_rates.get_value_at_time(t),
            ps=sampling_proportions.get_value_at_time(t),
        )
        for t in [0, *change_times]
    ]

    if notification_probability:
        notification_sampling_rate = skyline_parameter(notification_sampling_rate)
        notification_probability = skyline_parameter(notification_probability)
        models = [
            CTModel(
                model,
                phi=notification_sampling_rate.get_value_at_time(t),
                upsilon=notification_probability.get_value_at_time(t),
                allow_irremovable_states=allow_irremovable_states,
            )
            for t, model in zip([0, *change_times], models)
        ]

    [tree], _, _ = generate(
        models,
        min_tips=min_tips,
        max_tips=max_tips,
        T=T,
        skyline_times=change_times,
        state_frequencies=state_frequencies,
        max_notified_contacts=max_notified_contacts,
        root_state=root_state,
        random_seed=random_seed,
    )
    for i, leaf in enumerate(tree.iter_leaves()):
        state: str = getattr(leaf, STATE)
        date = tree.get_distance(leaf)
        leaf.name = f"{i}|{state}|{date}"
    save_forest([tree], output_file)


def get_BD_params(
    reproduction_number: SkylineParameterLike,
    infectious_period: SkylineParameterLike,
    sampling_proportion: SkylineParameterLike,
) -> TreeParams:
    transmission_rate = reproduction_number / infectious_period
    removal_rate = 1 / infectious_period
    return TreeParams(
        populations=INFECTIOUS_POPULATION,
        transmission_rates=transmission_rate,
        removal_rates=removal_rate,
        sampling_proportions=sampling_proportion,
    )


def get_BDEI_params(
    reproduction_number: SkylineParameterLike,
    infectious_period: SkylineParameterLike,
    incubation_period: SkylineParameterLike,
    sampling_proportion: SkylineParameterLike,
) -> TreeParams:
    transmission_rates = [[0, 0], [reproduction_number / infectious_period, 0]]
    transition_rates = [[0, 1 / incubation_period], [0, 0]]
    removal_rates = [0, 1 / infectious_period]
    sampling_proportions = [0, sampling_proportion]
    return TreeParams(
        populations=[EXPOSED_POPULATION, INFECTIOUS_POPULATION],
        transition_rates=transition_rates,
        transmission_rates=transmission_rates,
        removal_rates=removal_rates,
        sampling_proportions=sampling_proportions,
    )
