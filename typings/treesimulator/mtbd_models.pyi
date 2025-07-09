from collections.abc import Sequence

import numpy as np

class Model(object):
    def __init__(
        self,
        states: Sequence[str] | None = None,
        transition_rates: Sequence[Sequence[float]] | None = None,
        transmission_rates: Sequence[Sequence[float]] | None = None,
        removal_rates: Sequence[float] | None = None,
        ps: Sequence[float] | None = None,
        state_frequencies: Sequence[float] | None = None,
        n_recipients: Sequence[float] | None = None,
    ) -> None: ...

class CTModel(Model):
    def __init__(
        self,
        model: Model,
        phi: float = np.inf,
        upsilon: float = 0.5,
        allow_irremovable_states: bool = False,
    ) -> None: ...
