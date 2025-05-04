from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

from kitpy import flatten_dict
from kitpy.type_hints import Vector

from phylogenie.skyline import SkylineMatrix, SkylineParameter, SkylineVector


@dataclass
class Rates:
    birth_rates: SkylineVector
    death_rates: SkylineVector
    sampling_rates: SkylineVector
    migration_rates: SkylineMatrix
    birth_rates_among_demes: SkylineMatrix


@dataclass(kw_only=True, frozen=True)
class Parameterization(ABC):
    populations: list[str] = field(default_factory=lambda: ["X"])

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


@dataclass(frozen=True)
class CanonicalParameterization(Parameterization):
    death_rates: SkylineVector
    sampling_rates: SkylineVector
    birth_rates: SkylineVector | None = None
    migration_rates: SkylineMatrix | None = None
    birth_rates_among_demes: SkylineMatrix | None = None

    @cached_property
    def rates(self) -> Rates:
        return Rates(
            birth_rates=self._init_optional_vector(self.birth_rates),
            death_rates=self.death_rates,
            sampling_rates=self.sampling_rates,
            migration_rates=self._init_optional_matrix(self.migration_rates),
            birth_rates_among_demes=self._init_optional_matrix(
                self.birth_rates_among_demes
            ),
        )

    def serialize(self) -> dict[str, Vector]:
        rates = {
            "death_rates": self.death_rates.serialize(self.populations),
            "sampling_rates": self.sampling_rates.serialize(self.populations),
        }
        if self.birth_rates is not None:
            rates["birth_rates"] = self.birth_rates.serialize(self.populations)
        if self.migration_rates is not None:
            rates["migration_rates"] = self.migration_rates.serialize(self.populations)
        if self.birth_rates_among_demes is not None:
            rates["birth_rates_among_demes"] = self.birth_rates_among_demes.serialize(
                self.populations
            )
        return flatten_dict(rates)


@dataclass(frozen=True)
class EpidemiologicalParameterization(Parameterization):
    become_uninfectious_rates: SkylineVector
    sampling_proportions: SkylineVector
    reproduction_numbers: SkylineVector | None = None
    migration_rates: SkylineMatrix | None = None
    reproduction_numbers_among_demes: SkylineMatrix | None = None

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


@dataclass(kw_only=True, frozen=True)
class BDParameterization(Parameterization):
    populations: list[str] = field(default_factory=lambda: ["I"])
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


@dataclass(kw_only=True, frozen=True)
class BDEIParameterization(Parameterization):
    populations: list[str] = field(default_factory=lambda: ["E", "I"])
    reproduction_number: SkylineParameter
    sampling_proportion: SkylineParameter

    @property
    @abstractmethod
    def infectious_period_(self) -> SkylineParameter: ...

    @property
    @abstractmethod
    def incubation_period_(self) -> SkylineParameter: ...

    @cached_property
    def rates(self) -> Rates:
        return EpidemiologicalParameterization(
            populations=self.populations,
            become_uninfectious_rates=SkylineVector([0, 1 / self.infectious_period_]),
            sampling_proportions=SkylineVector([0, self.sampling_proportion]),
            migration_rates=SkylineMatrix([[1 / self.incubation_period_], [0]]),
            reproduction_numbers_among_demes=SkylineMatrix(
                [[0], [self.reproduction_number]]
            ),
        ).rates

    def serialize(self) -> dict[str, Vector]:
        return flatten_dict(
            {
                "reproduction_number": self.reproduction_number.serialize(),
                "infectious_period": self.infectious_period_.serialize(),
                "incubation_period": self.incubation_period_.serialize(),
                "sampling_proportion": self.sampling_proportion.serialize(),
            }
        )


@dataclass(frozen=True)
class CanonicalBDEIParameterization(BDEIParameterization):
    infectious_period: SkylineParameter
    incubation_period: SkylineParameter

    @property
    def infectious_period_(self) -> SkylineParameter:
        return self.infectious_period

    @property
    def incubation_period_(self) -> SkylineParameter:
        return self.incubation_period


@dataclass(frozen=True)
class IncubationFractionBDEIParameterization(BDEIParameterization):
    infection_period: SkylineParameter
    incubation_fraction: SkylineParameter

    @cached_property
    def infectious_period_(self) -> SkylineParameter:
        return self.infection_period * (1 - self.incubation_fraction)

    @cached_property
    def incubation_period_(self) -> SkylineParameter:
        return self.infection_period * self.incubation_fraction
