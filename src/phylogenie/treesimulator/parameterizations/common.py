from dataclasses import dataclass

from phylogenie.treesimulator.core import Model, StochasticEvent, TimedEvent


@dataclass(kw_only=True)
class StateDependentEvent(StochasticEvent):
    state: str | None = None

    def get_propensity(self, model: Model) -> float:
        rate = self.rate.get_value_at_time(model.current_time)
        return rate * model.count_active_nodes(self.state)

    def draw(self, model: Model):
        return model.draw_active_node(self.state)


class Death(StateDependentEvent):
    def apply(self, model: Model):
        model.remove(self.draw(model))


Recovery = Death


@dataclass(kw_only=True)
class Migration(StateDependentEvent):
    target_state: str

    def apply(self, model: Model):
        model.migrate(self.draw(model), self.target_state)


@dataclass(kw_only=True)
class Sampling(StateDependentEvent):
    removal: bool

    def apply(self, model: Model):
        model.sample(self.draw(model), self.removal)


@dataclass(kw_only=True)
class TimedSampling(TimedEvent):
    state: str | None = None
    proportion: float
    removal: bool

    def apply(self, model: Model):
        for node in model.get_active_nodes(self.state):
            if model.rng.random() < self.proportion:
                model.sample(node, self.removal)
