from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy

from numpy.random import Generator

from phylogenie.skyline import SkylineParameterLike, skyline_parameter
from phylogenie.treesimulator.events.base import Event, EventType
from phylogenie.treesimulator.events.core import Birth, Death, Migration, Sampling
from phylogenie.treesimulator.model import Model

CT_POSTFIX = "-CT"
CONTACTS_KEY = "CONTACTS"


def get_CT_state(state: str) -> str:
    return f"{state}{CT_POSTFIX}"


def is_CT_state(state: str) -> bool:
    return state.endswith(CT_POSTFIX)


class BirthWithContactTracing(Event):
    type = EventType.BIRTH

    def __init__(self, state: str, rate: SkylineParameterLike, child_state: str):
        super().__init__(state, rate)
        self.child_state = child_state

    def apply(self, model: Model, events: list[Event], time: float, rng: Generator):
        individual = self.draw_individual(model, rng)
        new_individual = model.birth_from(individual, self.child_state, time)
        if CONTACTS_KEY not in model.metadata:
            model[CONTACTS_KEY] = defaultdict(list)
        model[CONTACTS_KEY][individual].append(new_individual)
        model[CONTACTS_KEY][new_individual].append(individual)

    def __repr__(self) -> str:
        return f"BirthWithContactTracing(state={self.state}, rate={self.rate}, child_state={self.child_state})"


class SamplingWithContactTracing(Event):
    type = EventType.SAMPLING

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

    def apply(self, model: Model, events: list[Event], time: float, rng: Generator):
        individual = self.draw_individual(model, rng)
        model.sample(individual, time, True)
        population = model.get_population()
        if CONTACTS_KEY not in model.metadata:
            return
        contacts = model[CONTACTS_KEY][individual]
        for contact in contacts[-self.max_notified_contacts :]:
            if contact in population:
                state = model.get_state(contact)
                p = self.notification_probability.get_value_at_time(time)
                if not is_CT_state(state) and rng.random() < p:
                    model.migrate(contact, get_CT_state(state), time)

    def __repr__(self) -> str:
        return f"SamplingWithContactTracing(state={self.state}, rate={self.rate}, max_notified_contacts={self.max_notified_contacts}, notification_probability={self.notification_probability})"


def get_contact_tracing_events(
    events: Sequence[Event],
    max_notified_contacts: int = 1,
    notification_probability: SkylineParameterLike = 0.0,
    sampling_rate_after_notification: SkylineParameterLike = 2**32,
    samplable_states_after_notification: list[str] | None = None,
) -> list[Event]:
    ct_events: list[Event] = []
    notification_probability = skyline_parameter(notification_probability)
    sampling_rate_after_notification = skyline_parameter(
        sampling_rate_after_notification
    )
    for event in [deepcopy(e) for e in events]:
        if isinstance(event, Migration):
            ct_events.append(event)
            ct_events.append(
                Migration(
                    get_CT_state(event.state),
                    event.rate,
                    get_CT_state(event.target_state),
                )
            )
        elif isinstance(event, Birth):
            ct_events.append(
                BirthWithContactTracing(event.state, event.rate, event.child_state)
            )
            ct_events.append(
                BirthWithContactTracing(
                    get_CT_state(event.state), event.rate, event.child_state
                )
            )
        elif isinstance(event, Sampling):
            if not event.removal:
                raise ValueError(
                    "Contact tracing requires removal to be set for all sampling events."
                )
            ct_events.append(
                SamplingWithContactTracing(
                    event.state,
                    event.rate,
                    max_notified_contacts,
                    notification_probability,
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
                get_CT_state(state),
                sampling_rate_after_notification,
                max_notified_contacts,
                notification_probability,
            )
        )

    return ct_events
