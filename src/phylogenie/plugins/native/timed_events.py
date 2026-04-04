from abc import ABC, abstractmethod
from typing import Annotated, Any

from numpy.random import Generator

import phylogenie.generators.configs as cfg
import phylogenie.generators.factories as f
import phylogenie.treesimulator as ts
from phylogenie.utils import Registry


class TimedEvent(ABC, cfg.ForbidExtraBaseModel):
    time: cfg.Scalar
    firings: cfg.Scalar

    @abstractmethod
    def fn_factory(
        self, context: dict[str, Any], rng: Generator
    ) -> ts.TimedEventFunction: ...

    def factory(self, context: dict[str, Any], rng: Generator) -> ts.TimedEvent:
        time = f.scalar(self.time, context)
        firings = f.scalar(self.firings, context)
        return ts.TimedEvent(
            time=time, firings=firings, fn=self.fn_factory(context, rng)
        )


TIMED_EVENT_REGISTRY = Registry(TimedEvent)


@TIMED_EVENT_REGISTRY.register("sampling")
class TimedSamplingModel(TimedEvent):
    state: str | None = None
    removal: bool

    def fn_factory(self, context: dict[str, Any], rng: Generator):
        return ts.Sampling(state=self.state, removal=self.removal)


@TIMED_EVENT_REGISTRY.register("death")
class TimedDeathModel(TimedEvent):
    state: str | None = None

    def fn_factory(self, context: dict[str, Any], rng: Generator):
        return ts.Death(state=self.state)


TimedEventConfig = Annotated[TimedEvent, TIMED_EVENT_REGISTRY.validator]
