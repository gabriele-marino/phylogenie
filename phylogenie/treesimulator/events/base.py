from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator

from phylogenie.skyline import SkylineParameterLike, skyline_parameter
from phylogenie.treesimulator.model import Model


class Event(ABC):
    def __init__(self, state: str, rate: SkylineParameterLike):
        self.state = state
        self.rate = skyline_parameter(rate)

    def draw_individual(self, model: Model, rng: Generator) -> int:
        return rng.choice(model.get_population(self.state))

    def get_propensity(self, model: Model, time: float) -> float:
        n_individuals = model.count_individuals(self.state)
        rate = self.rate.get_value_at_time(time)
        if rate == np.inf and not n_individuals:
            return 0
        return rate * n_individuals

    @abstractmethod
    def apply(self, model: Model, time: float, rng: Generator) -> None: ...
