from abc import ABC, abstractmethod
from functools import cached_property

from pydantic import BaseModel, Field
from pykit import flatten_dict
from pykit.type_hints import Vector

from phylogenie.skyline import SkylineMatrix, SkylineParameter, SkylineVector


class Rates(BaseModel):
    birth_rates: SkylineVector
    death_rates: SkylineVector
    sampling_rates: SkylineVector
    migration_rates: SkylineMatrix
    birth_rates_among_demes: SkylineMatrix
    ancestor_sampling_rates: SkylineVector


class Parameterization(ABC, BaseModel):
    populations: list[str] = Field(default_factory=lambda: ["X"])

    @property
    @abstractmethod
    def rates(self) -> Rates: ...

    @abstractmethod
    def serialize(self) -> dict[str, Vector]: ...

    def _init_optional_vector(self, vector: SkylineVector | None) -> SkylineVector:
        if vector is None:
            N = len(self.populations)
            return SkylineVector([0] * N)
        return vector

    def _init_optional_matrix(self, matrix: SkylineMatrix | None) -> SkylineMatrix:
        if matrix is None:
            N = len(self.populations)
            return SkylineMatrix([[0] * (N - 1)] * N)
        return matrix


class CanonicalParameterization(Parameterization):
    death_rates: SkylineVector
    sampling_rates: SkylineVector | None = None
    birth_rates: SkylineVector | None = None
    migration_rates: SkylineMatrix | None = None
    birth_rates_among_demes: SkylineMatrix | None = None
    ancestor_sampling_rates: SkylineVector | None = None

    @cached_property
    def rates(self) -> Rates:
        return Rates(
            birth_rates=self._init_optional_vector(self.birth_rates),
            death_rates=self.death_rates,
            sampling_rates=self._init_optional_vector(self.sampling_rates),
            migration_rates=self._init_optional_matrix(self.migration_rates),
            birth_rates_among_demes=self._init_optional_matrix(
                self.birth_rates_among_demes
            ),
            ancestor_sampling_rates=self._init_optional_vector(
                self.ancestor_sampling_rates
            ),
        )

    def serialize(self) -> dict[str, Vector]:
        rates = {"death_rates": self.death_rates.serialize(self.populations)}
        if self.sampling_rates is not None:
            rates["sampling_rates"] = self.sampling_rates.serialize(self.populations)
        if self.birth_rates is not None:
            rates["birth_rates"] = self.birth_rates.serialize(self.populations)
        if self.migration_rates is not None:
            rates["migration_rates"] = self.migration_rates.serialize(self.populations)
        if self.birth_rates_among_demes is not None:
            rates["birth_rates_among_demes"] = self.birth_rates_among_demes.serialize(
                self.populations
            )
        return flatten_dict(rates)


class EpidemiologicalParameterization(Parameterization):
    become_uninfectious_rates: SkylineVector
    sampling_proportions: SkylineVector
    reproduction_numbers: SkylineVector | None = None
    migration_rates: SkylineMatrix | None = None
    reproduction_numbers_among_demes: SkylineMatrix | None = None
    ancestor_sampling_rates: SkylineVector | None = None

    @cached_property
    def rates(self) -> Rates:
        return Rates(
            birth_rates=(
                self._init_optional_vector(self.reproduction_numbers)
                * self.become_uninfectious_rates
            ),
            death_rates=self.become_uninfectious_rates
            * (1 - self.sampling_proportions),
            sampling_rates=self.become_uninfectious_rates * self.sampling_proportions,
            migration_rates=self._init_optional_matrix(self.migration_rates),
            birth_rates_among_demes=self._init_optional_matrix(
                self.reproduction_numbers_among_demes
            )
            * self.become_uninfectious_rates,
            ancestor_sampling_rates=self._init_optional_vector(
                self.ancestor_sampling_rates
            ),
        )

    def serialize(self) -> dict[str, Vector]:
        rates = {
            "become_uninfectious_rates": self.become_uninfectious_rates.serialize(
                self.populations
            ),
            "sampling_proportions": self.sampling_proportions.serialize(
                self.populations
            ),
        }
        if self.reproduction_numbers is not None:
            rates["reproduction_numbers"] = self.reproduction_numbers.serialize(
                self.populations
            )
        if self.migration_rates is not None:
            rates["migration_rates"] = self.migration_rates.serialize(self.populations)
        if self.reproduction_numbers_among_demes is not None:
            rates["reproduction_numbers_among_demes"] = (
                self.reproduction_numbers_among_demes.serialize(self.populations)
            )
        return flatten_dict(rates)


class BDParameterization(Parameterization):
    populations: list[str] = Field(default_factory=lambda: ["I"])
    reproduction_number: SkylineParameter
    infectious_period: SkylineParameter
    sampling_proportion: SkylineParameter

    @cached_property
    def rates(self) -> Rates:
        return EpidemiologicalParameterization(
            populations=self.populations,
            reproduction_numbers=SkylineVector([self.reproduction_number]),
            become_uninfectious_rates=SkylineVector([1 / self.infectious_period]),
            sampling_proportions=SkylineVector([self.sampling_proportion]),
        ).rates

    def serialize(self) -> dict[str, Vector]:
        return flatten_dict(
            {
                "reproduction_number": self.reproduction_number.serialize(),
                "infectious_period": self.infectious_period.serialize(),
                "sampling_proportion": self.sampling_proportion.serialize(),
            }
        )


class BDEIParameterization(Parameterization):
    populations: list[str] = Field(default_factory=lambda: ["E", "I"])
    reproduction_number: SkylineParameter
    infectious_period: SkylineParameter
    incubation_period: SkylineParameter
    sampling_proportion: SkylineParameter

    @cached_property
    def rates(self) -> Rates:
        return EpidemiologicalParameterization(
            populations=self.populations,
            become_uninfectious_rates=SkylineVector([0, 1 / self.infectious_period]),
            sampling_proportions=SkylineVector([0, self.sampling_proportion]),
            migration_rates=SkylineMatrix([[1 / self.incubation_period], [0]]),
            reproduction_numbers_among_demes=SkylineMatrix(
                [[0], [self.reproduction_number]]
            ),
        ).rates

    def serialize(self) -> dict[str, Vector]:
        return flatten_dict(
            {
                "reproduction_number": self.reproduction_number.serialize(),
                "infectious_period": self.infectious_period.serialize(),
                "incubation_period": self.incubation_period.serialize(),
                "sampling_proportion": self.sampling_proportion.serialize(),
            }
        )


class FractionalBDEIParameterization(BDEIParameterization):
    def __init__(
        self,
        infection_period: SkylineParameter,
        incubation_fraction: SkylineParameter,
        reproduction_number: SkylineParameter,
        sampling_proportion: SkylineParameter,
        populations: list[str] | None = None,
    ):
        if populations is None:
            populations = ["E", "I"]
        infectious_period = infection_period * (1 - incubation_fraction)
        incubation_period = infection_period * incubation_fraction

        super().__init__(
            populations=populations,
            infectious_period=infectious_period,
            incubation_period=incubation_period,
            reproduction_number=reproduction_number,
            sampling_proportion=sampling_proportion,
        )
