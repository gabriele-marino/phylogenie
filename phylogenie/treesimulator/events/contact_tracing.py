from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from numpy.random import Generator

from phylogenie.skyline import SkylineParameterLike, skyline_parameter
from phylogenie.treesimulator.events.core import Birth, Death, Migration, Sampling
from phylogenie.treesimulator.model import Event, Model

CT_POSTFIX = "-CT"
CONTACTS_KEY = "CONTACTS"


def _get_CT_state(state: str) -> str:
    return f"{state}{CT_POSTFIX}"


def _is_CT_state(state: str) -> bool:
    return state.endswith(CT_POSTFIX)


class BirthWithContactTracing(Event):
    def __init__(self, state: str, rate: SkylineParameterLike, child_state: str):
        super().__init__(state, rate)
        self.child_state = child_state

    def apply(self, model: Model, time: float, rng: Generator) -> None:
        individual = self.draw_individual(model, rng)
        new_individual = model.birth_from(individual, self.child_state, time)
        if CONTACTS_KEY not in model.context:
            model.context[CONTACTS_KEY] = defaultdict(list)
        model.context[CONTACTS_KEY][individual].append(new_individual)
        model.context[CONTACTS_KEY][new_individual].append(individual)

    def __repr__(self) -> str:
        return f"BirthWithContactTracing(state={self.state}, rate={self.rate}, child_state={self.child_state})"


class SamplingWithContactTracing(Event):
    def __init__(
        self,
        state: str,
        rate: SkylineParameterLike,
        max_notified_contacts: int,
        notification_probability: SkylineParameterLike,
    ):
        super().__init__(state, rate)
        self.max_notified_contacts = max_notified_contacts
        self.notification_probability = skyline_parameter(notification_probability)

    def apply(self, model: Model, time: float, rng: Generator) -> None:
        individual = self.draw_individual(model, rng)
        model.sample(individual, time, True)
        population = model.get_population()
        if CONTACTS_KEY not in model.context:
            return
        contacts = model.context[CONTACTS_KEY][individual]
        for contact in contacts[-self.max_notified_contacts :]:
            if contact in population:
                state = model.get_state(contact)
                p = self.notification_probability.get_value_at_time(time)
                if not _is_CT_state(state) and rng.random() < p:
                    model.migrate(contact, _get_CT_state(state), time)

    def __repr__(self) -> str:
        return f"SamplingWithContactTracing(state={self.state}, rate={self.rate}, max_notified_contacts={self.max_notified_contacts}, notification_probability={self.notification_probability})"


def get_contact_tracing_events(
    events: Sequence[Event],
    max_notified_contacts: int = 1,
    notification_probability: SkylineParameterLike = 1,
    sampling_rate_after_notification: SkylineParameterLike = np.inf,
    samplable_states_after_notification: list[str] | None = None,
) -> list[Event]:
    ct_events: list[Event] = []
    notification_probability = skyline_parameter(notification_probability)
    sampling_rate_after_notification = skyline_parameter(
        sampling_rate_after_notification
    )
    for event in events:
        state, rate = event.state, event.rate
        if isinstance(event, Migration):
            ct_events.append(event)
            ct_events.append(
                Migration(_get_CT_state(state), rate, _get_CT_state(event.target_state))
            )
        elif isinstance(event, Birth):
            ct_events.append(BirthWithContactTracing(state, rate, event.child_state))
            ct_events.append(
                BirthWithContactTracing(_get_CT_state(state), rate, event.child_state)
            )
        elif isinstance(event, Sampling):
            if not event.removal:
                raise ValueError(
                    "Contact tracing requires removal to be set for all sampling events."
                )
            ct_events.append(
                SamplingWithContactTracing(
                    state, rate, max_notified_contacts, notification_probability
                )
            )
        elif isinstance(event, Death):
            ct_events.append(event)
        else:
            raise NotImplementedError(
                f"Unsupported event type {type(event)} for contact tracing."
            )

    for state in (
        samplable_states_after_notification
        if samplable_states_after_notification is not None
        else {e.state for e in events}
    ):
        ct_events.append(
            SamplingWithContactTracing(
                _get_CT_state(state),
                sampling_rate_after_notification,
                max_notified_contacts,
                notification_probability,
            )
        )

    return ct_events
