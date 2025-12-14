import re
from copy import deepcopy
from typing import Any, Callable

from numpy.random import Generator

from phylogenie.skyline import SkylineParameterLike
from phylogenie.treesimulator.events.base import Event, EventType
from phylogenie.treesimulator.events.contact_tracing import (
    BirthWithContactTracing,
    SamplingWithContactTracing,
)
from phylogenie.treesimulator.events.core import Birth, Death, Migration, Sampling
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


class Mutation(Event):
    type = EventType.MUTATION

    def __init__(
        self,
        state: str,
        rate: SkylineParameterLike,
        rate_scalers: dict[EventType, Callable[[], float]],
        rates_to_log: list[EventType] | None = None,
    ):
        super().__init__(state, rate)
        self.rate_scalers = rate_scalers
        self.rates_to_log = [] if rates_to_log is None else rates_to_log

    def apply(
        self, model: Model, events: list[Event], time: float, rng: Generator
    ) -> dict[str, Any]:
        if NEXT_MUTATION_ID not in model.metadata:
            model[NEXT_MUTATION_ID] = 0
        model[NEXT_MUTATION_ID] += 1
        mutation_id = model[NEXT_MUTATION_ID]

        individual = self.draw_individual(model, rng)
        model.migrate(individual, _get_mutated_state(mutation_id, self.state), time)

        rate_scalers: dict[EventType, float] = {
            target_type: rate_scaler()
            for target_type, rate_scaler in self.rate_scalers.items()
        }

        metadata: dict[str, Any] = {}
        for event in [
            deepcopy(e)
            for e in events
            if _get_mutation(self.state) == _get_mutation(e.state)
        ]:
            event.state = _get_mutated_state(mutation_id, event.state)

            if isinstance(event, Birth | BirthWithContactTracing):
                event.child_state = _get_mutated_state(mutation_id, event.child_state)
                metadata_key = f"birth_rate_from_{event.state}_to_{event.child_state}"
            elif isinstance(event, Migration):
                event.target_state = _get_mutated_state(mutation_id, event.target_state)
                metadata_key = (
                    f"migration_rate_from_{event.state}_to_{event.target_state}"
                )
            elif isinstance(
                event, Mutation | Death | Sampling | SamplingWithContactTracing
            ):
                metadata_key = f"{event.type}_rate_for_{event.state}"
            else:
                raise ValueError(
                    f"Mutation not implemented for event of type {type(event)}."
                )

            event.rate *= rate_scalers.get(event.type, 1)
            if event.type in self.rates_to_log:
                metadata[metadata_key] = (
                    event.rate.value[0]
                    if len(event.rate.value) == 1
                    else list(event.rate.value)
                )

            events.append(event)

        return metadata

    def __repr__(self) -> str:
        return f"Mutation(state={self.state}, rate={self.rate}, rate_scalers={self.rate_scalers})"
