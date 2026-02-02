from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import comb
from typing import Generic, TypeVar

from numpy.random import Generator

from phylogenie.skyline import SkylineParameterLike, skyline_parameter
from phylogenie.treesimulator.models import Model

M = TypeVar("M", bound="Model")


class Event(ABC, Generic[M]):
    def __init__(self, reactants: dict[str, int], rate: SkylineParameterLike):
        self.reactants = reactants
        self.rate = skyline_parameter(rate)
        if any(v < 0 for v in self.rate.value):
            raise ValueError("Event rates must be non-negative.")

    def draw_reactants(self, model: M, rng: Generator) -> dict[str, list[int]]:
        return {
            state: rng.choice(
                model.get_individuals(state), size=count, replace=False
            ).tolist()
            for state, count in self.reactants.items()
        }

    def get_propensity(self, model: M, time: float) -> float:
        propensity = self.rate.get_value_at_time(time)
        for state, count in self.reactants.items():
            propensity *= comb(model.count_individuals(state), count)
        return propensity

    @abstractmethod
    def apply(self, model: M, time: float, rng: Generator): ...


class SingleReactantEvent(Event[M]):
    def __init__(self, state: str, rate: SkylineParameterLike):
        super().__init__({state: 1}, rate)
        self.state = state

    def draw_individual(self, model: M, rng: Generator) -> int:
        return rng.choice(model.get_individuals(self.state))


class TimedEvent(ABC, Generic[M]):
    def __init__(self, times: Sequence[float]):
        if any(t < 0 for t in times):
            raise ValueError("Event times must be non-negative.")
        self.times = sorted(times)

    @abstractmethod
    def apply(self, model: M, time: float, rng: Generator): ...
