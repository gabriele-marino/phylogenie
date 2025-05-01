from typing import Any

class Model(object):
    def __init__(
        self,
        states: list[str] | None = None,
        transition_rates: list[list[float]] | None = None,
        transmission_rates: list[list[float]] | None = None,
        removal_rates: list[float] | None = None,
        ps: list[float] | None = None,
        state_frequencies: list[float] | None = None,
        n_recipients: list[float] | None = None,
        *args: Any,
        **kwargs: Any
    ) -> None: ...
