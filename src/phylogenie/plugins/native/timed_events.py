from abc import ABC, abstractmethod
from typing import Annotated, Any

from numpy.random import Generator
from pydantic import BeforeValidator

import phylogenie.generators.configs as cfg
import phylogenie.generators.factories as f
import phylogenie.treesimulator.parameterizations as pmt
from phylogenie.utils import Registry


class TimedEvent(ABC, cfg.StrictBaseModel):
    times: cfg.ManyScalars
    firings: cfg.Scalar

    @abstractmethod
    def fn_factory(
        self, context: dict[str, Any], rng: Generator
    ) -> pmt.TimedEventFunction: ...

    def factory(self, context: dict[str, Any], rng: Generator) -> pmt.TimedEvent:
        times = f.many_scalars(self.times, context)
        firings = f.scalar(self.firings, context)
        return pmt.TimedEvent(
            times=times, firings=firings, fn=self.fn_factory(context, rng)
        )


TimedEventRegistry = Registry(TimedEvent)


@TimedEventRegistry.register("sampling")
class TimedSamplingModel(TimedEvent):
    state: str | None = None
    removal: bool

    def fn_factory(self, context: dict[str, Any], rng: Generator):
        return pmt.Sampling(state=self.state, removal=self.removal)


@TimedEventRegistry.register("death")
class TimedDeathModel(TimedEvent):
    state: str | None = None

    def fn_factory(self, context: dict[str, Any], rng: Generator):
        return pmt.Death(state=self.state)


TimedEventConfig = Annotated[
    TimedEvent, BeforeValidator(lambda v: TimedEventRegistry.load(v))
]
