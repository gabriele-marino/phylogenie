from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from numpy.random import Generator

from phylogenie.skyline import SkylineParameterLike, skyline_parameter
from phylogenie.treesimulator.model import Model


class EventType(str, Enum):
    BIRTH = "birth"
    DEATH = "death"
    MIGRATION = "migration"
    SAMPLING = "sampling"
    MUTATION = "mutation"


class Event(ABC):
    type: EventType

    def __init__(self, state: str, rate: SkylineParameterLike):
        self.state = state
        self.rate = skyline_parameter(rate)
        if any(v < 0 for v in self.rate.value):
            raise ValueError("Event rates must be non-negative.")

    def draw_individual(self, model: Model, rng: Generator) -> int:
        return rng.choice(model.get_population(self.state))

    def get_propensity(self, model: Model, time: float) -> float:
        n_individuals = model.count_individuals(self.state)
        rate = self.rate.get_value_at_time(time)
        return rate * n_individuals

    @abstractmethod
    def apply(
        self, model: Model, events: "list[Event]", time: float, rng: Generator
    ) -> dict[str, Any] | None: ...
