from typing import Any

from treesimulator import STATE, save_forest
from treesimulator.generator import generate
from treesimulator.mtbd_models import Model

from phylogenie.parameterizations import Rates
from phylogenie.skyline import SkylineVector


def generate_tree(
    rates: Rates,
    populations: list[str],
    output_file: str,
    **kwargs: Any,
) -> None:
    populations = populations
    rates = rates
    become_uninfectious_rates = rates.death_rates + rates.sampling_rates
    sampling_proportions = SkylineVector(
        [
            sampling_rate / become_uninfectious_rate if become_uninfectious_rate else 0
            for sampling_rate, become_uninfectious_rate in zip(
                rates.sampling_rates, become_uninfectious_rates
            )
        ]
    )

    change_times = sorted(
        set(
            [
                0,
                *rates.birth_rates.change_times,
                *rates.death_rates.change_times,
                *rates.sampling_rates.change_times,
                *rates.migration_rates.change_times,
                *rates.birth_rates_among_demes.change_times,
            ]
        )
    )

    models = []
    for t in change_times:
        transition_rates = rates.migration_rates.get_value_at_time(t)
        for i, population_transition_rates in enumerate(transition_rates):
            population_transition_rates.insert(i, 0)

        birth_rates = rates.birth_rates.get_value_at_time(t)
        transmission_rates = rates.birth_rates_among_demes.get_value_at_time(t)
        for i, (birth_rate, population_transition_rates) in enumerate(
            zip(birth_rates, transmission_rates)
        ):
            population_transition_rates.insert(i, birth_rate)

        models.append(
            Model(
                states=populations,
                transition_rates=transition_rates,
                transmission_rates=transmission_rates,
                removal_rates=become_uninfectious_rates.get_value_at_time(t),
                sampling_probabilities=sampling_proportions.get_value_at_time(t),
            )
        )

    [tree], _, _ = generate(models, skyline_times=change_times, **kwargs)
    for i, leaf in enumerate(tree.iter_leaves()):
        state = getattr(leaf, STATE).capitalize()
        date = tree.get_distance(leaf)
        leaf.name = f"{i}|{state}|{date}"
    save_forest([tree], output_file)
