from phylogenie.skyline import SkylineParameterLike, skyline_parameter
from phylogenie.treesimulator.core import Model, StochasticEvent
from phylogenie.treesimulator.parameterizations.common import Recovery, Sampling

INFECTIOUS_STATE = "I"
SUSCEPTIBLE_STATE = "S"

SUSCEPTIBLES_KEY = "susceptibles"


class Transmission(StochasticEvent):
    def get_propensity(self, model: Model) -> float:
        rate = self.rate.get_value_at_time(model.current_time)
        susceptibles = model[SUSCEPTIBLES_KEY]
        return rate * model.count_active_nodes(INFECTIOUS_STATE) * susceptibles

    def apply(self, model: Model):
        model[SUSCEPTIBLES_KEY] -= 1
        parent_node = model.draw_active_node(INFECTIOUS_STATE)
        model.birth_from(parent_node, SUSCEPTIBLE_STATE)


def get_SIR_model(
    transmission_rate: SkylineParameterLike,
    recovery_rate: SkylineParameterLike,
    sampling_rate: SkylineParameterLike,
    susceptibles: int,
):
    model = Model(
        init_state=INFECTIOUS_STATE, init_metadata={SUSCEPTIBLES_KEY: susceptibles}
    )
    model.add_stochastic_event(Transmission(rate=skyline_parameter(transmission_rate)))
    model.add_stochastic_event(
        Recovery(rate=skyline_parameter(recovery_rate), state=INFECTIOUS_STATE)
    )
    model.add_stochastic_event(
        Sampling(
            rate=skyline_parameter(sampling_rate), state=INFECTIOUS_STATE, removal=True
        )
    )
    return model
