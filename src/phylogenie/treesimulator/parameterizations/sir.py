from phylogenie.skyline import SkylineParameterLike, skyline_parameter
from phylogenie.treesimulator.core import Model
from phylogenie.treesimulator.parameterizations.core import (
    Recovery,
    Sampling,
    StochasticEvent,
)

INFECTIOUS_STATE = "I"
SUSCEPTIBLE_STATE = "S"

SUSCEPTIBLES = "susceptibles"


class Transmission:
    def reactant_combinations(self, model: Model) -> int:
        return model[SUSCEPTIBLES] * model.count_active_nodes(INFECTIOUS_STATE)

    def apply(self, model: Model):
        model[SUSCEPTIBLES] -= 1
        parent_node = model.draw_active_node(INFECTIOUS_STATE)
        model.birth_from(parent_node, INFECTIOUS_STATE)


def get_SIR_model(
    transmission_rate: SkylineParameterLike,
    recovery_rate: SkylineParameterLike,
    sampling_rate: SkylineParameterLike,
    susceptibles: int,
):
    model = Model(
        init_state=INFECTIOUS_STATE, init_metadata={SUSCEPTIBLES: susceptibles}
    )
    model.add_event(
        StochasticEvent(rate=skyline_parameter(transmission_rate), fn=Transmission())
    )
    model.add_event(
        StochasticEvent(
            rate=skyline_parameter(recovery_rate), fn=Recovery(state=INFECTIOUS_STATE)
        )
    )
    model.add_event(
        StochasticEvent(
            rate=skyline_parameter(sampling_rate),
            fn=Sampling(state=INFECTIOUS_STATE, removal=True),
        )
    )
    return model
