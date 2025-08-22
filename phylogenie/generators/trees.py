from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from numpy.random import default_rng
from pydantic import Field

import phylogenie.generators.configs as cfg
from phylogenie.generators.dataset import DatasetGenerator, DataType
from phylogenie.generators.factories import (
    data,
    distribution,
    integer,
    scalar,
    skyline_matrix,
    skyline_parameter,
    skyline_vector,
)
from phylogenie.io import dump_newick
from phylogenie.tree import Tree
from phylogenie.treesimulator import (
    Event,
    Mutation,
    get_BD_events,
    get_BDEI_events,
    get_BDSS_events,
    get_canonical_events,
    get_contact_tracing_events,
    get_epidemiological_events,
    get_FBD_events,
    simulate_tree,
)


class ParameterizationType(str, Enum):
    CANONICAL = "canonical"
    EPIDEMIOLOGICAL = "epidemiological"
    FBD = "FBD"
    BD = "BD"
    BDEI = "BDEI"
    BDSS = "BDSS"


class TreeDatasetGenerator(DatasetGenerator):
    data_type: Literal[DataType.TREES] = DataType.TREES
    min_tips: cfg.Integer = 1
    max_tips: cfg.Integer = 2**32
    max_time: cfg.Scalar = np.inf
    init_state: str | None = None
    sampling_probability_at_present: cfg.Scalar = 0.0
    timeout: float = np.inf

    @abstractmethod
    def _get_events(self, data: dict[str, Any]) -> list[Event]: ...

    def simulate_one(self, data: dict[str, Any], seed: int | None = None) -> Tree:
        init_state = (
            self.init_state
            if self.init_state is None
            else self.init_state.format(**data)
        )
        return simulate_tree(
            events=self._get_events(data),
            min_tips=integer(self.min_tips, data),
            max_tips=integer(self.max_tips, data),
            max_time=scalar(self.max_time, data),
            init_state=init_state,
            sampling_probability_at_present=scalar(
                self.sampling_probability_at_present, data
            ),
            seed=seed,
            timeout=self.timeout,
        )

    def generate_one(
        self,
        filename: str,
        context: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        d = {"file_id": Path(filename).stem}
        rng = default_rng(seed)
        while True:
            try:
                d.update(data(context, rng))
                tree = self.simulate_one(d, seed)
                dump_newick(tree, f"{filename}.nwk")
                break
            except TimeoutError:
                print("Simulation timed out, retrying with different parameters...")
        return d


class CanonicalTreeDatasetGenerator(TreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.CANONICAL] = (
        ParameterizationType.CANONICAL
    )
    states: list[str]
    sampling_rates: cfg.SkylineVector
    remove_after_sampling: bool
    birth_rates: cfg.SkylineVector = 0
    death_rates: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    birth_rates_among_states: cfg.SkylineMatrix = None

    def _get_events(self, data: dict[str, Any]) -> list[Event]:
        return get_canonical_events(
            states=self.states,
            sampling_rates=skyline_vector(self.sampling_rates, data),
            remove_after_sampling=self.remove_after_sampling,
            birth_rates=skyline_vector(self.birth_rates, data),
            death_rates=skyline_vector(self.death_rates, data),
            migration_rates=skyline_matrix(self.migration_rates, data),
            birth_rates_among_states=skyline_matrix(
                self.birth_rates_among_states, data
            ),
        )


class FBDTreeDatasetGenerator(TreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.FBD] = ParameterizationType.FBD
    states: list[str]
    sampling_proportions: cfg.SkylineVector
    diversification: cfg.SkylineVector = 0
    turnover: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    diversification_between_states: cfg.SkylineMatrix = None

    def _get_events(self, data: dict[str, Any]) -> list[Event]:
        return get_FBD_events(
            states=self.states,
            diversification=skyline_vector(self.diversification, data),
            turnover=skyline_vector(self.turnover, data),
            sampling_proportions=skyline_vector(self.sampling_proportions, data),
            migration_rates=skyline_matrix(self.migration_rates, data),
            diversification_between_states=skyline_matrix(
                self.diversification_between_states, data
            ),
        )


class TreeDatasetGeneratorForEpidemiology(TreeDatasetGenerator):
    max_notified_contacts: cfg.Integer = 1
    notification_probability: cfg.SkylineParameter = 0.0
    sampling_rate_after_notification: cfg.SkylineParameter = np.inf
    samplable_states_after_notification: list[str] | None = None
    mutations: tuple[cfg.Mutation, ...] = Field(default_factory=tuple)

    @abstractmethod
    def _get_base_events(self, data: dict[str, Any]) -> list[Event]: ...

    def _get_events(self, data: dict[str, Any]) -> list[Event]:
        events = self._get_base_events(data)
        if self.notification_probability:
            events = get_contact_tracing_events(
                events=events,
                max_notified_contacts=integer(self.max_notified_contacts, data),
                notification_probability=skyline_parameter(
                    self.notification_probability, data
                ),
                sampling_rate_after_notification=skyline_parameter(
                    self.sampling_rate_after_notification, data
                ),
                samplable_states_after_notification=self.samplable_states_after_notification,
            )
        all_states = list({e.state for e in events})
        for mutation in self.mutations:
            states = mutation.states
            if isinstance(states, str):
                states = [states]
            elif states is None:
                states = all_states
            for state in states:
                if state not in all_states:
                    raise ValueError(
                        f"Mutation state '{state}' is not found in states {all_states}."
                    )
                rate_scalers = {
                    t: distribution(r, data) for t, r in mutation.rate_scalers.items()
                }
                events.append(
                    Mutation(
                        state, skyline_parameter(mutation.rate, data), rate_scalers
                    )
                )
        return events


class EpidemiologicalTreeDatasetGenerator(TreeDatasetGeneratorForEpidemiology):
    parameterization: Literal[ParameterizationType.EPIDEMIOLOGICAL] = (
        ParameterizationType.EPIDEMIOLOGICAL
    )
    states: list[str]
    sampling_proportions: cfg.SkylineVector
    reproduction_numbers: cfg.SkylineVector = 0
    become_uninfectious_rates: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    reproduction_numbers_among_states: cfg.SkylineMatrix = None

    def _get_base_events(self, data: dict[str, Any]) -> list[Event]:
        return get_epidemiological_events(
            states=self.states,
            reproduction_numbers=skyline_vector(self.reproduction_numbers, data),
            become_uninfectious_rates=skyline_vector(
                self.become_uninfectious_rates, data
            ),
            sampling_proportions=skyline_vector(self.sampling_proportions, data),
            migration_rates=skyline_matrix(self.migration_rates, data),
            reproduction_numbers_among_states=skyline_matrix(
                self.reproduction_numbers_among_states, data
            ),
        )


class BDTreeDatasetGenerator(TreeDatasetGeneratorForEpidemiology):
    parameterization: Literal[ParameterizationType.BD] = ParameterizationType.BD
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter = 1

    def _get_base_events(self, data: dict[str, Any]) -> list[Event]:
        return get_BD_events(
            reproduction_number=skyline_parameter(self.reproduction_number, data),
            infectious_period=skyline_parameter(self.infectious_period, data),
            sampling_proportion=skyline_parameter(self.sampling_proportion, data),
        )


class BDEITreeDatasetGenerator(TreeDatasetGeneratorForEpidemiology):
    parameterization: Literal[ParameterizationType.BDEI] = ParameterizationType.BDEI
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    incubation_period: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter = 1

    def _get_base_events(self, data: dict[str, Any]) -> list[Event]:
        return get_BDEI_events(
            reproduction_number=skyline_parameter(self.reproduction_number, data),
            infectious_period=skyline_parameter(self.infectious_period, data),
            incubation_period=skyline_parameter(self.incubation_period, data),
            sampling_proportion=skyline_parameter(self.sampling_proportion, data),
        )


class BDSSTreeDatasetGenerator(TreeDatasetGeneratorForEpidemiology):
    parameterization: Literal[ParameterizationType.BDSS] = ParameterizationType.BDSS
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    superspreading_ratio: cfg.SkylineParameter
    superspreaders_proportion: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter = 1

    def _get_base_events(self, data: dict[str, Any]) -> list[Event]:
        return get_BDSS_events(
            reproduction_number=skyline_parameter(self.reproduction_number, data),
            infectious_period=skyline_parameter(self.infectious_period, data),
            superspreading_ratio=skyline_parameter(self.superspreading_ratio, data),
            superspreaders_proportion=skyline_parameter(
                self.superspreaders_proportion, data
            ),
            sampling_proportion=skyline_parameter(self.sampling_proportion, data),
        )


TreeDatasetGeneratorConfig = Annotated[
    CanonicalTreeDatasetGenerator
    | EpidemiologicalTreeDatasetGenerator
    | FBDTreeDatasetGenerator
    | BDTreeDatasetGenerator
    | BDEITreeDatasetGenerator
    | BDSSTreeDatasetGenerator,
    Field(discriminator="parameterization"),
]
