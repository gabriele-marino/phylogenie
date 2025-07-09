from collections.abc import Iterable
from enum import Enum
from typing import Annotated, Literal

from numpy.random import Generator
from pydantic import Field

import phylogenie.core.configs as cfg
from phylogenie.backend.remaster import (
    DEFAULT_POPULATION,
    SAMPLE_POPULATION,
    Reaction,
    generate_trees,
    get_canonical_reactions,
    get_epidemiological_reactions,
    get_FBD_reactions,
)
from phylogenie.core.factories import (
    skyline_matrix_like_factory,
    skyline_vector_like_factory,
)
from phylogenie.core.trees.base import BackendType, TreesGenerator
from phylogenie.core.trees.remaster.configs import (
    PunctualReactionConfig,
    ReactionConfig,
)
from phylogenie.core.trees.remaster.factories import (
    punctual_reaction_factory,
    reaction_factory,
)
from phylogenie.core.typings import Data


class ParameterizationType(str, Enum):
    CANONICAL = "canonical"
    EPIDEMIOLOGICAL = "epidemiological"
    FBD = "fbd"


class ReMASTERGenerator(TreesGenerator):
    backend: Literal[BackendType.REMASTER] = BackendType.REMASTER
    populations: str | list[str] = DEFAULT_POPULATION
    init_population: str = DEFAULT_POPULATION
    sample_population: str = SAMPLE_POPULATION
    reactions: Iterable[ReactionConfig] = Field(default_factory=tuple)
    punctual_reactions: Iterable[PunctualReactionConfig] = Field(default_factory=tuple)
    trajectory_attrs: dict[str, str | int | float] = Field(default_factory=dict)

    def _generate_one_from_extra_reactions(
        self, filename: str, rng: Generator, data: Data, reactions: list[Reaction]
    ) -> None:
        generate_trees(
            tree_file_name=f"{filename}.nwk",
            populations=self.populations,
            init_population=self.init_population.format(**data),
            sample_population=self.sample_population,
            reactions=[reaction_factory(r, data) for r in self.reactions] + reactions,
            punctual_reactions=[
                punctual_reaction_factory(r, data) for r in self.punctual_reactions
            ],
            trajectory_attrs={
                k: v.format(**data) if isinstance(v, str) else str(v)
                for k, v in self.trajectory_attrs.items()
            },
            seed=int(rng.integers(0, 2**31 - 1)),
        )

    def _generate_one(self, filename: str, rng: Generator, data: Data) -> None:
        self._generate_one_from_extra_reactions(filename, rng, data, reactions=[])


class CanonicalReMASTERGenerator(ReMASTERGenerator):
    parameterization: Literal[ParameterizationType.CANONICAL] = (
        ParameterizationType.CANONICAL
    )
    birth_rates: cfg.SkylineVectorLikeConfig = 0
    death_rates: cfg.SkylineVectorLikeConfig = 0
    sampling_rates: cfg.SkylineVectorLikeConfig = 0
    removal_probabilities: cfg.SkylineVectorLikeConfig = 0
    migration_rates: cfg.SkylineMatrixLikeConfig = 0
    birth_rates_among_demes: cfg.SkylineMatrixLikeConfig = 0

    def _generate_one(self, filename: str, rng: Generator, data: Data) -> None:
        reactions = get_canonical_reactions(
            populations=self.populations,
            sample_population=self.sample_population,
            birth_rates=skyline_vector_like_factory(self.birth_rates, data),
            death_rates=skyline_vector_like_factory(self.death_rates, data),
            sampling_rates=skyline_vector_like_factory(self.sampling_rates, data),
            removal_probabilities=skyline_vector_like_factory(
                self.removal_probabilities, data
            ),
            migration_rates=skyline_matrix_like_factory(self.migration_rates, data),
            birth_rates_among_demes=skyline_matrix_like_factory(
                self.birth_rates_among_demes, data
            ),
        )
        self._generate_one_from_extra_reactions(filename, rng, data, reactions)


class EpidemiologicalReMASTERGenerator(ReMASTERGenerator):
    parameterization: Literal[ParameterizationType.EPIDEMIOLOGICAL] = (
        ParameterizationType.EPIDEMIOLOGICAL
    )
    reproduction_numbers: cfg.SkylineVectorLikeConfig = 0
    become_uninfectious_rates: cfg.SkylineVectorLikeConfig = 0
    sampling_proportions: cfg.SkylineVectorLikeConfig = 0
    removal_probabilities: cfg.SkylineVectorLikeConfig = 0
    migration_rates: cfg.SkylineMatrixLikeConfig = 0
    reproduction_numbers_among_demes: cfg.SkylineMatrixLikeConfig = 0

    def _generate_one(self, filename: str, rng: Generator, data: Data) -> None:
        reactions = get_epidemiological_reactions(
            populations=self.populations,
            sample_population=self.sample_population,
            reproduction_numbers=skyline_vector_like_factory(
                self.reproduction_numbers, data
            ),
            become_uninfectious_rates=skyline_vector_like_factory(
                self.become_uninfectious_rates, data
            ),
            sampling_proportions=skyline_vector_like_factory(
                self.sampling_proportions, data
            ),
            removal_probabilities=skyline_vector_like_factory(
                self.removal_probabilities, data
            ),
            migration_rates=skyline_matrix_like_factory(self.migration_rates, data),
            reproduction_numbers_among_demes=skyline_matrix_like_factory(
                self.reproduction_numbers_among_demes, data
            ),
        )
        self._generate_one_from_extra_reactions(filename, rng, data, reactions)


class FBDReMASTERGenerator(ReMASTERGenerator):
    parameterization: Literal[ParameterizationType.FBD] = ParameterizationType.FBD
    diversification: cfg.SkylineVectorLikeConfig = 0
    turnover: cfg.SkylineVectorLikeConfig = 0
    sampling_proportions: cfg.SkylineVectorLikeConfig = 0
    removal_probabilities: cfg.SkylineVectorLikeConfig = 0
    migration_rates: cfg.SkylineMatrixLikeConfig = 0
    diversification_between_types: cfg.SkylineMatrixLikeConfig = 0

    def _generate_one(self, filename: str, rng: Generator, data: Data) -> None:
        reactions = get_FBD_reactions(
            populations=self.populations,
            sample_population=self.sample_population,
            diversification=skyline_vector_like_factory(self.diversification, data),
            turnover=skyline_vector_like_factory(self.turnover, data),
            sampling_proportions=skyline_vector_like_factory(
                self.sampling_proportions, data
            ),
            removal_probabilities=skyline_vector_like_factory(
                self.removal_probabilities, data
            ),
            migration_rates=skyline_matrix_like_factory(self.migration_rates, data),
            diversification_between_types=skyline_matrix_like_factory(
                self.diversification_between_types, data
            ),
        )
        self._generate_one_from_extra_reactions(filename, rng, data, reactions)


ReMASTERGeneratorConfig = Annotated[
    CanonicalReMASTERGenerator
    | EpidemiologicalReMASTERGenerator
    | FBDReMASTERGenerator,
    Field(discriminator="parameterization"),
]
