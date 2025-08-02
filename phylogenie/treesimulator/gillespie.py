from collections.abc import Sequence

import numpy as np
from numpy.random import default_rng

from phylogenie.skyline import SkylineParameterLike
from phylogenie.tree import Tree
from phylogenie.treesimulator.events import Event, get_contact_tracing_events
from phylogenie.treesimulator.model import Model, is_CT_state


def simulate_tree(
    events: Sequence[Event],
    min_tips: int = 1,
    max_tips: int | None = None,
    max_time: float = np.inf,
    init_state: str | None = None,
    sampling_probability_at_present: float = 0.0,
    notification_probability: float = 0,
    max_notified_contacts: int = 1,
    samplable_states_after_notification: Sequence[str] | None = None,
    sampling_rate_after_notification: SkylineParameterLike = np.inf,
    contacts_removal_probability: SkylineParameterLike = 1,
    max_tries: int | None = None,
    seed: int | None = None,
) -> Tree | None:
    rng = default_rng(seed)

    if max_tips is None and max_time == np.inf:
        raise ValueError("Either max_tips or max_time must be specified.")

    if notification_probability:
        events = get_contact_tracing_events(
            events,
            samplable_states_after_notification,
            sampling_rate_after_notification,
            contacts_removal_probability,
        )

    n_tries = 0
    root_states = [e.state for e in events if not is_CT_state(e.state)]
    while max_tries is None or n_tries < max_tries:
        root_state = init_state if init_state is not None else rng.choice(root_states)
        model = Model(root_state, max_notified_contacts, notification_probability, rng)
        current_time = 0.0
        change_times = sorted(set(t for e in events for t in e.rate.change_times))
        next_change_time = change_times.pop(0) if change_times else np.inf
        n_tips = None if max_tips is None else rng.integers(min_tips, max_tips + 1)

        while current_time < max_time and (n_tips is None or model.n_sampled < n_tips):
            rates = [e.get_propensity(model, current_time) for e in events]

            instantaneous_events = [e for e, r in zip(events, rates) if r == np.inf]
            if instantaneous_events:
                event = instantaneous_events[rng.integers(len(instantaneous_events))]
                event.apply(model, current_time)
                continue

            if not any(rates):
                break

            current_time += rng.exponential(1 / sum(rates))
            if current_time >= max_time:
                break

            if current_time >= next_change_time:
                current_time = next_change_time
                next_change_time = change_times.pop(0) if change_times else np.inf
                continue

            event_idx = np.searchsorted(np.cumsum(rates) / sum(rates), rng.random())
            events[int(event_idx)].apply(model, current_time)

        for individual in model.get_population():
            if rng.random() < sampling_probability_at_present:
                model.sample(individual, current_time, 1)

        if model.n_sampled >= min_tips and (
            max_tips is None or model.n_sampled <= max_tips
        ):
            return model.get_sampled_tree()

        n_tries += 1
