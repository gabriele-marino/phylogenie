from collections.abc import Callable
from dataclasses import dataclass

from phylogenie.skyline import SkylineParameter

Event = Callable[[float], None]


@dataclass
class StochasticEvent:
    fn: Event
    reactants: dict[str, int]
    rate: SkylineParameter


@dataclass
class TimedEvent:
    fn: Event
    times: list[float]
