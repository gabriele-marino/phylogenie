import re
from copy import deepcopy
from enum import Enum
from typing import Any, Type

from numpy.random import Generator

from phylogenie.models import Distribution
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

MUTATION_PREFIX = "MUT-"
NEXT_MUTATION_ID = "NEXT_MUTATION_ID"


def _get_mutation(state: str) -> str | None:
    return state.split(".")[0] if state.startswith(MUTATION_PREFIX) else None


def _get_mutated_state(mutation_id: int, state: str) -> str:
    if state.startswith(MUTATION_PREFIX):
        _, state = state.split(".")
    return f"{MUTATION_PREFIX}{mutation_id}.{state}"


def get_mutation_id(node_name: str) -> int:
    match = re.search(rf"{MUTATION_PREFIX}(\d+)\.", node_name)
    if match:
        return int(match.group(1))
    return 0


class TargetType(str, Enum):
    BIRTH = "birth"
    DEATH = "death"
    MIGRATION = "migration"
    SAMPLING = "sampling"


EVENT_TARGET_TYPES: dict[Type[Event], TargetType] = {
    Birth: TargetType.BIRTH,
    BirthWithContactTracing: TargetType.BIRTH,
    Death: TargetType.DEATH,
    Migration: TargetType.MIGRATION,
    Sampling: TargetType.SAMPLING,
    SamplingWithContactTracing: TargetType.SAMPLING,
}


class Mutation:
    def __init__(self, rate: float, rate_scalers: dict[TargetType, Distribution]):
        self.rate = rate
        self.rate_scalers = rate_scalers

    def apply(
        self, model: Model, events: list[Event], time: float, rng: Generator
    ) -> dict[str, Any]:
        if NEXT_MUTATION_ID not in model.context:
            model.context[NEXT_MUTATION_ID] = 0
        model.context[NEXT_MUTATION_ID] += 1
        mutation_id = model.context[NEXT_MUTATION_ID]

        individual = rng.choice(model.get_population())
        state = model.get_state(individual)
        model.migrate(individual, _get_mutated_state(mutation_id, state), time)

        rate_scalers: dict[TargetType, float] = {
            target_type: getattr(rng, rate_scaler.type)(**rate_scaler.args)
            for target_type, rate_scaler in self.rate_scalers.items()
        }

        metadata: dict[str, Any] = {}
        for event in [
            deepcopy(e)
            for e in events
            if _get_mutation(state) == _get_mutation(e.state)
        ]:
            event.state = _get_mutated_state(mutation_id, event.state)
            if isinstance(event, Birth | BirthWithContactTracing):
                event.child_state = _get_mutated_state(mutation_id, event.child_state)
            elif isinstance(event, Migration):
                event.target_state = _get_mutated_state(mutation_id, event.target_state)
            elif not isinstance(event, Death | Sampling | SamplingWithContactTracing):
                raise ValueError(
                    f"Mutation not implemented for event of type {type(event)}."
                )

            target_type = EVENT_TARGET_TYPES[type(event)]
            if target_type in rate_scalers:
                event.rate *= rate_scalers[target_type]
                metadata[f"{MUTATION_PREFIX}{mutation_id}.{target_type}.rate.value"] = (
                    event.rate.value[0]
                    if len(event.rate.value) == 1
                    else list(event.rate.value)
                )

            events.append(event)

        return metadata

    def __repr__(self) -> str:
        return f"Mutation(rate={self.rate}, rate_scalers={self.rate_scalers})"
