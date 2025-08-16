from copy import deepcopy
from enum import Enum
from typing import Type

from numpy.random import Generator

from phylogenie.skyline import SkylineParameterLike
from phylogenie.treesimulator.events.contact_tracing import (
    BirthWithContactTracing,
    SamplingWithContactTracing,
)
from phylogenie.treesimulator.events.core import (
    Birth,
    Death,
    Event,
    Migration,
    Sampling,
)
from phylogenie.treesimulator.model import Model
from phylogenie.utils import Distribution

MUTATION_PREFIX = "MUT-"
MUTATIONS_KEY = "MUTATIONS"


def _get_mutation(state: str) -> str | None:
    return state.split(".")[0] if state.startswith(MUTATION_PREFIX) else None


def _get_mutated_state(mutation_id: int, state: str) -> str:
    if state.startswith(MUTATION_PREFIX):
        state = state.split(".")[1]
    return f"{MUTATION_PREFIX}{mutation_id}.{state}"


class TargetType(str, Enum):
    BIRTH = "birth"
    DEATH = "death"
    MIGRATION = "migration"
    SAMPLING = "sampling"
    MUTATION = "mutation"


class Mutation(Event):
    def __init__(
        self,
        state: str,
        rate: SkylineParameterLike,
        rate_scalers: dict[TargetType, Distribution],
    ):
        super().__init__(state, rate)
        self.rate_scalers = rate_scalers

    def apply(self, model: Model, time: float, rng: Generator) -> None:
        if MUTATIONS_KEY not in model.context:
            model.context[MUTATIONS_KEY] = 0
        model.context[MUTATIONS_KEY] += 1
        mutation_id = model.context[MUTATIONS_KEY]

        individual = self.draw_individual(model, rng)
        model.migrate(individual, _get_mutated_state(mutation_id, self.state), time)

        rate_scalers = {
            target_type: getattr(rng, rate_scaler.type)(**rate_scaler.args)
            for target_type, rate_scaler in self.rate_scalers.items()
        }

        for event in [
            deepcopy(e)
            for e in model.events
            if _get_mutation(self.state) == _get_mutation(e.state)
        ]:
            event.state = _get_mutated_state(mutation_id, event.state)
            if isinstance(event, Birth | BirthWithContactTracing):
                event.child_state = _get_mutated_state(mutation_id, event.child_state)
            elif isinstance(event, Migration):
                event.target_state = _get_mutated_state(mutation_id, event.target_state)
            elif not isinstance(
                event, Mutation | Death | Sampling | SamplingWithContactTracing
            ):
                raise ValueError(
                    f"Mutation not defined for event of type {type(event)}."
                )

            for target_type, rate_scaler in rate_scalers.items():
                if target_type not in TARGETS:
                    raise ValueError(
                        f"Unsupported target type {target_type} for mutation."
                    )
                if isinstance(event, TARGETS[target_type]):
                    event.rate *= rate_scaler

            model.add_event(event)

    def __repr__(self) -> str:
        return f"Mutation(state={self.state}, rate={self.rate}, rate_scalers={self.rate_scalers})"


TARGETS: dict[TargetType, tuple[Type[Event], ...]] = {
    TargetType.BIRTH: (Birth, BirthWithContactTracing),
    TargetType.DEATH: (Death,),
    TargetType.MIGRATION: (Migration,),
    TargetType.SAMPLING: (Sampling, SamplingWithContactTracing),
    TargetType.MUTATION: (Mutation,),
}
