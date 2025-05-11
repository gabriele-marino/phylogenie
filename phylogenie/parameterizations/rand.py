from abc import ABC
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pykit.validators import EnsuredList

from phylogenie.parameterizations.core import (
    BDEIParameterization,
    BDParameterization,
    CanonicalParameterization,
    EpidemiologicalParameterization,
    FractionalBDEIParameterization,
)
from phylogenie.skyline import (
    RandomSkylineMatrix,
    RandomSkylineParameter,
    RandomSkylineVector,
    SkylineMatrix,
    SkylineVector,
)


class ParameterizationType(str, Enum):
    CANONICAL = "canonical"
    EPIDEMIOLOGICAL = "epidemiological"
    BD = "BD"
    BDEI = "BDEI"
    INCUBATION_FRACTION_BDEI = "fractional-BDEI"


class BaseRandomParameterization(ABC, BaseModel):
    type: ParameterizationType
    populations: EnsuredList[str] = Field(default_factory=lambda: ["X"])

    def _sample_optional_vector(
        self, vector: RandomSkylineVector | None
    ) -> SkylineVector | None:
        if vector is None:
            return None
        N = len(self.populations)
        return vector.sample(N)

    def _sample_optional_matrix(
        self, matrix: RandomSkylineMatrix | None
    ) -> SkylineMatrix | None:
        if matrix is None:
            return None
        N = len(self.populations)
        return matrix.sample(N)


class RandomCanonicalParameterization(BaseRandomParameterization):
    type: Literal[ParameterizationType.CANONICAL] = ParameterizationType.CANONICAL
    death_rates: RandomSkylineVector
    sampling_rates: RandomSkylineVector
    birth_rates: RandomSkylineVector | None = None
    migration_rates: RandomSkylineMatrix | None = None
    birth_rates_among_demes: RandomSkylineMatrix | None = None
    ancestor_sampling_rates: RandomSkylineVector | None = None

    def sample(self) -> CanonicalParameterization:
        N = len(self.populations)
        return CanonicalParameterization(
            populations=self.populations,
            birth_rates=self._sample_optional_vector(self.birth_rates),
            death_rates=self.death_rates.sample(N),
            sampling_rates=self.sampling_rates.sample(N),
            migration_rates=self._sample_optional_matrix(self.migration_rates),
            birth_rates_among_demes=self._sample_optional_matrix(
                self.birth_rates_among_demes
            ),
            ancestor_sampling_rates=self._sample_optional_vector(
                self.ancestor_sampling_rates
            ),
        )


class RandomEpidemiologicalParameterization(BaseRandomParameterization):
    type: Literal[ParameterizationType.EPIDEMIOLOGICAL] = (
        ParameterizationType.EPIDEMIOLOGICAL
    )
    reproduction_numbers: RandomSkylineVector | None = None
    become_uninfectious_rates: RandomSkylineVector
    sampling_proportions: RandomSkylineVector
    migration_rates: RandomSkylineMatrix | None = None
    reproduction_numbers_among_demes: RandomSkylineMatrix | None = None
    ancestor_sampling_rates: RandomSkylineVector | None = None

    def sample(self) -> EpidemiologicalParameterization:
        N = len(self.populations)
        return EpidemiologicalParameterization(
            populations=self.populations,
            reproduction_numbers=self._sample_optional_vector(
                self.reproduction_numbers
            ),
            become_uninfectious_rates=self.become_uninfectious_rates.sample(N),
            sampling_proportions=self.sampling_proportions.sample(N),
            migration_rates=self._sample_optional_matrix(self.migration_rates),
            reproduction_numbers_among_demes=self._sample_optional_matrix(
                self.reproduction_numbers_among_demes
            ),
            ancestor_sampling_rates=self._sample_optional_vector(
                self.ancestor_sampling_rates
            ),
        )


class RandomBDParameterization(BaseRandomParameterization):
    type: Literal[ParameterizationType.BD] = ParameterizationType.BD
    populations: EnsuredList[str] = Field(default_factory=lambda: ["I"])
    reproduction_number: RandomSkylineParameter
    infectious_period: RandomSkylineParameter
    sampling_proportion: RandomSkylineParameter

    def sample(self) -> BDParameterization:
        return BDParameterization(
            populations=self.populations,
            reproduction_number=self.reproduction_number.sample(),
            infectious_period=self.infectious_period.sample(),
            sampling_proportion=self.sampling_proportion.sample(),
        )


class BaseRandomBDEIParameterization(BaseRandomParameterization):
    populations: EnsuredList[str] = Field(default_factory=lambda: ["E", "I"])
    reproduction_number: RandomSkylineParameter
    sampling_proportion: RandomSkylineParameter


class RandomBDEIParameterization(BaseRandomBDEIParameterization):
    type: Literal[ParameterizationType.BDEI] = ParameterizationType.BDEI
    infectious_period: RandomSkylineParameter
    incubation_period: RandomSkylineParameter

    def sample(self) -> BDEIParameterization:
        return BDEIParameterization(
            populations=self.populations,
            reproduction_number=self.reproduction_number.sample(),
            infectious_period=self.infectious_period.sample(),
            incubation_period=self.infectious_period.sample(),
            sampling_proportion=self.sampling_proportion.sample(),
        )


class RandomFractionalBDEIParameterization(BaseRandomBDEIParameterization):
    type: Literal[ParameterizationType.INCUBATION_FRACTION_BDEI] = (
        ParameterizationType.INCUBATION_FRACTION_BDEI
    )
    infection_period: RandomSkylineParameter
    incubation_fraction: RandomSkylineParameter

    def sample(self) -> FractionalBDEIParameterization:
        return FractionalBDEIParameterization(
            populations=self.populations,
            reproduction_number=self.reproduction_number.sample(),
            infection_period=self.infection_period.sample(),
            incubation_fraction=self.incubation_fraction.sample(),
            sampling_proportion=self.sampling_proportion.sample(),
        )


RandomParameterization = Annotated[
    RandomCanonicalParameterization
    | RandomEpidemiologicalParameterization
    | RandomBDParameterization
    | RandomBDEIParameterization
    | RandomFractionalBDEIParameterization,
    Field(discriminator="type"),
]
