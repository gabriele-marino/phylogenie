from enum import Enum
from typing import Annotated, Literal

import numpy as np
from numpy.random import Generator
from pydantic import Field

import phylogenie.core.configs as cfg
import phylogenie.typings as pgt
from phylogenie.backend.treesimulator import (
    DEFAULT_POPULATION,
    TreeParams,
    generate_tree,
    get_BD_params,
    get_BDEI_params,
    get_BDSS_params,
)
from phylogenie.core.factories import (
    int_factory,
    scalar_factory,
    skyline_matrix_coercible_factory,
    skyline_parameter_like_factory,
    skyline_vector_coercible_factory,
)
from phylogenie.core.trees.base import BackendType, TreesGenerator


class ParameterizationType(str, Enum):
    MTBD = "MTBD"
    BD = "BD"
    BDEI = "BDEI"
    BDSS = "BDSS"


class TreeSimulatorGenerator(TreesGenerator):
    backend: Literal[BackendType.TREESIMULATOR] = BackendType.TREESIMULATOR
    min_tips: cfg.IntConfig
    max_tips: cfg.IntConfig
    T: cfg.ScalarConfig = np.inf
    root_state: str | None = None
    state_frequencies: list[float] | None = None
    notification_probability: cfg.SkylineParameterLikeConfig = 0
    notification_sampling_rate: cfg.SkylineParameterLikeConfig = np.inf
    allow_irremovable_states: bool = False
    max_notified_contacts: cfg.IntConfig = 1

    def _generate_one_from_params(
        self, filename: str, rng: Generator, data: pgt.Data, params: TreeParams
    ) -> None:
        root_state = (
            self.root_state
            if self.root_state is None
            else self.root_state.format(**data)
        )
        generate_tree(
            output_file=f"{filename}.nwk",
            params=params,
            min_tips=int_factory(self.min_tips, data),
            max_tips=int_factory(self.max_tips, data),
            T=scalar_factory(self.T, data),
            state_frequencies=self.state_frequencies,
            notification_probability=skyline_parameter_like_factory(
                self.notification_probability, data
            ),
            notification_sampling_rate=skyline_parameter_like_factory(
                self.notification_sampling_rate, data
            ),
            allow_irremovable_states=self.allow_irremovable_states,
            max_notified_contacts=int_factory(self.max_notified_contacts, data),
            root_state=root_state,
            random_seed=int(rng.integers(0, 2**31 - 1)),
        )


class MTBDTreeSimulatorGenerator(TreeSimulatorGenerator):
    parameterization: Literal[ParameterizationType.MTBD] = ParameterizationType.MTBD
    populations: str | list[str] = DEFAULT_POPULATION
    transmission_rates: cfg.SkylineMatrixCoercibleConfig
    removal_rates: cfg.SkylineVectorCoercibleConfig
    transition_rates: cfg.SkylineMatrixCoercibleConfig = 0
    sampling_proportions: cfg.SkylineVectorCoercibleConfig = 1

    def _generate_one(self, filename: str, rng: Generator, data: pgt.Data) -> None:
        self._generate_one_from_params(
            filename,
            rng,
            data,
            TreeParams(
                populations=self.populations,
                transition_rates=skyline_matrix_coercible_factory(
                    self.transition_rates, data
                ),
                transmission_rates=skyline_matrix_coercible_factory(
                    self.transmission_rates, data
                ),
                removal_rates=skyline_vector_coercible_factory(
                    self.removal_rates, data
                ),
                sampling_proportions=skyline_vector_coercible_factory(
                    self.sampling_proportions, data
                ),
            ),
        )


class BDTreeSimulatorGenerator(TreeSimulatorGenerator):
    parameterization: Literal[ParameterizationType.BD] = ParameterizationType.BD
    reproduction_number: cfg.SkylineParameterLikeConfig
    infectious_period: cfg.SkylineParameterLikeConfig
    sampling_proportion: cfg.SkylineParameterLikeConfig = 1

    def _generate_one(self, filename: str, rng: Generator, data: pgt.Data) -> None:
        self._generate_one_from_params(
            filename,
            rng,
            data,
            get_BD_params(
                reproduction_number=skyline_parameter_like_factory(
                    self.reproduction_number, data
                ),
                infectious_period=skyline_parameter_like_factory(
                    self.infectious_period, data
                ),
                sampling_proportion=skyline_parameter_like_factory(
                    self.sampling_proportion, data
                ),
            ),
        )


class BDEITreeSimulatorGenerator(TreeSimulatorGenerator):
    parameterization: Literal[ParameterizationType.BDEI] = ParameterizationType.BDEI
    reproduction_number: cfg.SkylineParameterLikeConfig
    infectious_period: cfg.SkylineParameterLikeConfig
    incubation_period: cfg.SkylineParameterLikeConfig
    sampling_proportion: cfg.SkylineParameterLikeConfig = 1

    def _generate_one(self, filename: str, rng: Generator, data: pgt.Data) -> None:
        self._generate_one_from_params(
            filename,
            rng,
            data,
            get_BDEI_params(
                reproduction_number=skyline_parameter_like_factory(
                    self.reproduction_number, data
                ),
                infectious_period=skyline_parameter_like_factory(
                    self.infectious_period, data
                ),
                incubation_period=skyline_parameter_like_factory(
                    self.incubation_period, data
                ),
                sampling_proportion=skyline_parameter_like_factory(
                    self.sampling_proportion, data
                ),
            ),
        )


class BDSSTreeSimulatorGenerator(TreeSimulatorGenerator):
    parameterization: Literal[ParameterizationType.BDSS] = ParameterizationType.BDSS
    reproduction_number: cfg.SkylineParameterLikeConfig
    infectious_period: cfg.SkylineParameterLikeConfig
    superspreading_ratio: cfg.SkylineParameterLikeConfig
    superspreaders_proportion: cfg.SkylineParameterLikeConfig
    sampling_proportion: cfg.SkylineParameterLikeConfig = 1

    def _generate_one(self, filename: str, rng: Generator, data: pgt.Data) -> None:
        self._generate_one_from_params(
            filename,
            rng,
            data,
            get_BDSS_params(
                reproduction_number=skyline_parameter_like_factory(
                    self.reproduction_number, data
                ),
                infectious_period=skyline_parameter_like_factory(
                    self.infectious_period, data
                ),
                superspreading_ratio=skyline_parameter_like_factory(
                    self.superspreading_ratio, data
                ),
                superspreaders_proportion=skyline_parameter_like_factory(
                    self.superspreaders_proportion, data
                ),
                sampling_proportion=skyline_parameter_like_factory(
                    self.sampling_proportion, data
                ),
            ),
        )


TreeSimulatorGeneratorConfig = Annotated[
    MTBDTreeSimulatorGenerator
    | BDTreeSimulatorGenerator
    | BDEITreeSimulatorGenerator
    | BDSSTreeSimulatorGenerator,
    Field(discriminator="parameterization"),
]
