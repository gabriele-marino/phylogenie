import re
from abc import ABC, abstractmethod
from collections import defaultdict
from math import comb

import numpy as np
from numpy.random import default_rng

from phylogenie._mixins import MetadataMixin
from phylogenie.tree_node import TreeNode
from phylogenie.treesimulator.events import StochasticEvent, TimedEvent


class Model(MetadataMixin, ABC):
    def __init__(self, max_time: float | None = None, seed: int | None = None):
        super().__init__()
        self.max_time = max_time
        self._rng = default_rng(seed)
        self._stochastic_events: list[StochasticEvent] = []
        self._timed_events: list[TimedEvent] = []

    def init(self):
        self.clear()
        self._current_time = 0.0
        self._next_individual_id = 0
        self._population: dict[str, set[int]] = defaultdict(set)
        self._states: dict[int, str] = {}

    def create_new_individual(self, state: str) -> int:
        self._next_individual_id += 1
        self._population[state].add(self._next_individual_id)
        self._states[self._next_individual_id] = state
        return self._next_individual_id

    def remove_individual(self, individual: int) -> None:
        state = self._states[individual]
        self._population[state].remove(individual)
        del self._states[individual]

    def migrate_individual(self, individual: int, new_state: str) -> None:
        old_state = self._states[individual]
        self._population[old_state].remove(individual)
        self._states[individual] = new_state
        self._population[new_state].add(individual)

    def get_individuals(self, state: str) -> list[int]:
        return [
            individual
            for s, individuals in self._population.items()
            if re.fullmatch(state, s) is not None
            for individual in individuals
        ]

    def count_individuals(self, state: str) -> int:
        return len(self.get_individuals(state))

    def draw_individual(self, state: str) -> int:
        return self._rng.choice(self.get_individuals(state))

    def draw_individuals(self, count: int, state: str) -> list[int]:
        return self._rng.choice(
            self.get_individuals(state), size=count, replace=False
        ).tolist()

    @property
    @abstractmethod
    def tree_size(self) -> int: ...

    @abstractmethod
    def get_tree(self) -> TreeNode | None: ...

    # -------------------------
    # Gillespie-related methods
    # -------------------------

    def _get_propensity(self, event: StochasticEvent, time: float) -> float:
        propensity = event.rate.get_value_at_time(time)
        for state, count in event.reactants.items():
            propensity *= comb(self.count_individuals(state), count)
        return propensity

    def _get_next_change_time(self) -> float | None:
        change_times = [
            t
            for e in self._stochastic_events
            for t in e.rate.change_times
            if t > self._current_time
        ]
        return min(change_times) if change_times else None

    def _get_next_stochastic_event(
        self,
    ) -> tuple[float, StochasticEvent] | tuple[None, None]:
        propensities = [
            self._get_propensity(e, self._current_time) for e in self._stochastic_events
        ]
        total_propensity = sum(propensities)
        if not total_propensity:
            return None, None
        next_time = self._current_time + self._rng.exponential(1 / total_propensity)
        event_idx = np.searchsorted(
            np.cumsum(propensities) / total_propensity, self._rng.random()
        )
        return next_time, self._stochastic_events[int(event_idx)]

    def _get_next_timed_events(
        self,
    ) -> tuple[float, list[TimedEvent]] | tuple[None, None]:
        times = [
            t for e in self._timed_events for t in e.times if t > self._current_time
        ]
        if not times:
            return None, None
        next_time = min(times)
        return next_time, [e for e in self._timed_events if next_time in e.times]

    def step(self) -> bool:
        stochastic_event_time, stochastic_event = self._get_next_stochastic_event()
        timed_events_time, timed_events = self._get_next_timed_events()
        next_change_time = self._get_next_change_time()

        # If there are no more events to process, we stop the simulation.
        if all(
            t is None
            for t in [next_change_time, timed_events_time, stochastic_event_time]
        ):
            return False

        # If the next event is beyond the max time, we stop the simulation.
        if self.max_time is not None and all(
            t is None or self.max_time < t
            for t in [
                stochastic_event_time,
                timed_events_time,
                next_change_time,
            ]
        ):
            return False

        # If the next change time is the earliest, we advance time to it.
        if next_change_time is not None and all(
            t is None or next_change_time <= t
            for t in [timed_events_time, stochastic_event_time]
        ):
            self._current_time = next_change_time

        # If the next timed events time is the earliest, we process them.
        if timed_events_time is not None and all(
            t is None or timed_events_time <= t
            for t in [next_change_time, stochastic_event_time]
        ):
            self._current_time = timed_events_time
            for event in timed_events:  # pyright: ignore
                event.fn(self._current_time)

        # If the next stochastic event time is the earliest, we process it.
        if stochastic_event_time is not None and all(
            t is None or stochastic_event_time <= t
            for t in [next_change_time, timed_events_time]
        ):
            self._current_time = stochastic_event_time
            stochastic_event.fn(self._current_time)  # pyright: ignore

        return True
