from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from phylogenie.skyline import SkylineParameter
from phylogenie.tree_node import TreeNode
from phylogenie.treesimulator.core import STATE, Event, Model


def _get_next_greater(times: Iterable[float], current_time: float) -> float | None:
    return next((v for v in times if v > current_time), None)


class StochasticEventFunction(Protocol):
    def reactant_combinations(self, model: Model) -> int: ...
    def apply(self, model: Model): ...


class TimedEventFunction(Protocol):
    def max_firings(self, model: Model) -> int: ...
    def apply_firings(self, model: Model, firings: int): ...


@dataclass
class StochasticEvent(Event):
    rate: SkylineParameter
    fn: StochasticEventFunction

    def apply(self, model: Model):
        if model.current_time not in self.rate.change_times:
            self.fn.apply(model)

    def get_next_firing_time(self, model: Model) -> float | None:
        next_change_time = _get_next_greater(self.rate.change_times, model.current_time)
        rate = self.rate.get_value_at_time(model.current_time)
        propensity = rate * self.fn.reactant_combinations(model)
        if not propensity:
            return next_change_time
        next_firing_time = model.current_time + model.rng.expovariate(propensity)
        return (
            next_firing_time
            if next_change_time is None
            else min(next_firing_time, next_change_time)
        )


@dataclass
class TimedEvent(Event):
    times: Iterable[float]
    firings: float | int
    fn: TimedEventFunction

    def apply(self, model: Model):
        max_firings = self.fn.max_firings(model)
        firings = (
            int(self.firings * max_firings)
            if isinstance(self.firings, float)
            else min(self.firings, max_firings)
        )
        self.fn.apply_firings(model, firings)

    def get_next_firing_time(self, model: Model) -> float | None:
        return _get_next_greater(self.times, model.current_time)


@dataclass(kw_only=True)
class SingleReactantEventFunction(ABC):
    state: str | None = None

    def draw_one(self, model: Model) -> TreeNode:
        return model.draw_active_node(self.state)

    def reactant_combinations(self, model: Model) -> int:
        return model.count_active_nodes(self.state)

    @abstractmethod
    def apply_to_node(self, model: Model, node: TreeNode): ...

    def max_firings(self, model: Model) -> int:
        return model.count_active_nodes(self.state)

    def apply_firings(self, model: Model, firings: int):
        active_nodes = model.get_active_nodes(self.state)
        for node in model.rng.sample(active_nodes, firings):
            self.apply_to_node(model, node)

    def apply(self, model: Model):
        self.apply_to_node(model, self.draw_one(model))


class Death(SingleReactantEventFunction):
    def apply_to_node(self, model: Model, node: TreeNode):
        return model.remove(node)


Recovery = Death


@dataclass(kw_only=True)
class Migration(SingleReactantEventFunction):
    target_state: str

    def apply_to_node(self, model: Model, node: TreeNode):
        model.migrate(node, self.target_state)


@dataclass(kw_only=True)
class Sampling(SingleReactantEventFunction):
    removal: bool

    def apply_to_node(self, model: Model, node: TreeNode):
        if self.removal:
            model.sample(node)
        else:
            _, sample_node = model.birth_from(node, node[STATE])
            model.sample(sample_node)
