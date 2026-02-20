from abc import abstractmethod
from enum import Enum
from typing import Annotated, Any, Callable, Literal

from pydantic import Field

import phylogenie.generators._configs as cfg
from phylogenie.generators._factories import (
    eval_expression,
    integer,
    scalar,
    skyline_matrix,
    skyline_parameter,
    skyline_vector,
    timed_event,
)
from phylogenie.generators.tree.base import Backend, TreeGenerator
from phylogenie.io import dump_newick
from phylogenie.tree_node import TreeNode
from phylogenie.treesimulator import (
    Model,
    get_BD_model,
    get_BDEI_model,
    get_BDSS_model,
    get_canonical_model,
    get_epidemiological_model,
    get_FBD_model,
    get_SIR_model,
    simulate_tree,
)
from phylogenie.treesimulator.parameterizations.open_population import (
    EXPOSED_STATE,
    INFECTIOUS_STATE,
    SUPERSPREADER_STATE,
)


class Parameterization(str, Enum):
    CANONICAL = "canonical"
    EPIDEMIOLOGICAL = "epidemiological"
    FBD = "FBD"
    BD = "BD"
    BDEI = "BDEI"
    BDSS = "BDSS"
    SIR = "SIR"


class PhylogenieTreeGenerator(TreeGenerator):
    backend: Literal[Backend.PHYLOGENIE] = Backend.PHYLOGENIE
    n_leaves: cfg.Integer | None = None
    max_time: cfg.Scalar | None = None
    timeout: float | None = None
    timed_events: Annotated[list[cfg.TimedEvent], Field(default_factory=list)]
    node_features: dict[str, str] | None = None
    acceptance_criterion: str | None = None
    logs: dict[str, str] | None = None

    @abstractmethod
    def _get_model(self, context: dict[str, Any]) -> Model: ...

    def generate(
        self, filename: str, context: dict[str, Any], seed: int | None = None
    ) -> dict[str, Any]:
        model = self._get_model(context)
        model.rng.seed(seed)

        for event in self.timed_events:
            model.add_timed_event(timed_event(event, context))

        max_time = None if self.max_time is None else scalar(self.max_time, context)

        acceptance_criterion: None | Callable[[TreeNode], bool] = (
            None
            if self.acceptance_criterion is None
            else lambda tree: eval_expression(
                self.acceptance_criterion,  # pyright: ignore
                context,
                {"tree": tree},
            )
        )
        logs: None | dict[str, Callable[[TreeNode], Any]] = (
            None
            if self.logs is None
            else {
                key: lambda tree, expr=expr: eval_expression(
                    expr, context, {"tree": tree}
                )
                for key, expr in self.logs.items()
            }
        )

        tree, metadata = simulate_tree(
            model=model,
            n_leaves=None if self.n_leaves is None else integer(self.n_leaves, context),
            max_time=max_time,
            timeout=self.timeout,
            acceptance_criterion=acceptance_criterion,
            logs=logs,
        )

        if self.node_features is not None:
            for name, feature in self.node_features.items():
                mapping = getattr(tree, feature)
                for node in tree:
                    node[name] = mapping[node]

        dump_newick(tree, f"{filename}.nwk")
        return metadata


class CanonicalGenerator(PhylogenieTreeGenerator):
    parameterization: Literal[Parameterization.CANONICAL] = Parameterization.CANONICAL
    states: list[str]
    init_state: str
    sampling_rates: cfg.SkylineVector = 0
    remove_after_sampling: bool = False
    birth_rates: cfg.SkylineVector = 0
    death_rates: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    birth_rates_among_states: cfg.SkylineMatrix = None

    def _get_model(self, context: dict[str, Any]) -> Model:
        return get_canonical_model(
            init_state=self.init_state.format(**context),
            states=self.states,
            sampling_rates=skyline_vector(self.sampling_rates, context),
            remove_after_sampling=self.remove_after_sampling,
            birth_rates=skyline_vector(self.birth_rates, context),
            death_rates=skyline_vector(self.death_rates, context),
            migration_rates=skyline_matrix(self.migration_rates, context),
            birth_rates_among_states=skyline_matrix(
                self.birth_rates_among_states, context
            ),
        )


class FBDGenerator(PhylogenieTreeGenerator):
    parameterization: Literal[Parameterization.FBD] = Parameterization.FBD
    states: list[str]
    init_state: str
    sampling_proportions: cfg.SkylineVector = 0
    diversification: cfg.SkylineVector = 0
    turnover: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    diversification_between_states: cfg.SkylineMatrix = None

    def _get_model(self, context: dict[str, Any]) -> Model:
        return get_FBD_model(
            init_state=self.init_state.format(**context),
            states=self.states,
            diversification=skyline_vector(self.diversification, context),
            turnover=skyline_vector(self.turnover, context),
            sampling_proportions=skyline_vector(self.sampling_proportions, context),
            migration_rates=skyline_matrix(self.migration_rates, context),
            diversification_between_states=skyline_matrix(
                self.diversification_between_states, context
            ),
        )


class EpidemiologicalGenerator(PhylogenieTreeGenerator):
    parameterization: Literal[Parameterization.EPIDEMIOLOGICAL] = (
        Parameterization.EPIDEMIOLOGICAL
    )
    states: list[str]
    init_state: str
    sampling_proportions: cfg.SkylineVector
    reproduction_numbers: cfg.SkylineVector = 0
    become_uninfectious_rates: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    reproduction_numbers_among_states: cfg.SkylineMatrix = None

    def _get_model(self, context: dict[str, Any]) -> Model:
        return get_epidemiological_model(
            init_state=self.init_state.format(**context),
            states=self.states,
            reproduction_numbers=skyline_vector(self.reproduction_numbers, context),
            become_uninfectious_rates=skyline_vector(
                self.become_uninfectious_rates, context
            ),
            sampling_proportions=skyline_vector(self.sampling_proportions, context),
            migration_rates=skyline_matrix(self.migration_rates, context),
            reproduction_numbers_among_states=skyline_matrix(
                self.reproduction_numbers_among_states, context
            ),
        )


class BDGenerator(PhylogenieTreeGenerator):
    parameterization: Literal[Parameterization.BD] = Parameterization.BD
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter

    def _get_model(self, context: dict[str, Any]) -> Model:
        return get_BD_model(
            reproduction_number=skyline_parameter(self.reproduction_number, context),
            infectious_period=skyline_parameter(self.infectious_period, context),
            sampling_proportion=skyline_parameter(self.sampling_proportion, context),
        )


class BDEIGenerator(PhylogenieTreeGenerator):
    parameterization: Literal[Parameterization.BDEI] = Parameterization.BDEI
    init_state: str = EXPOSED_STATE
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    incubation_period: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter

    def _get_model(self, context: dict[str, Any]) -> Model:
        init_state = self.init_state.format(**context)
        if init_state not in [EXPOSED_STATE, INFECTIOUS_STATE]:
            raise ValueError(
                f"Invalid init_state '{init_state}' for BDEI model. "
                f"It must be either '{EXPOSED_STATE}' or '{INFECTIOUS_STATE}'."
            )
        return get_BDEI_model(
            init_state=init_state,
            reproduction_number=skyline_parameter(self.reproduction_number, context),
            infectious_period=skyline_parameter(self.infectious_period, context),
            incubation_period=skyline_parameter(self.incubation_period, context),
            sampling_proportion=skyline_parameter(self.sampling_proportion, context),
        )


class BDSSGenerator(PhylogenieTreeGenerator):
    parameterization: Literal[Parameterization.BDSS] = Parameterization.BDSS
    init_state: str = INFECTIOUS_STATE
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    superspreading_ratio: cfg.SkylineParameter
    superspreaders_proportion: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter

    def _get_model(self, context: dict[str, Any]) -> Model:
        init_state = self.init_state.format(**context)
        if init_state not in [INFECTIOUS_STATE, SUPERSPREADER_STATE]:
            raise ValueError(
                f"Invalid init_state '{init_state}' for BDSS model. "
                f"It must be either '{INFECTIOUS_STATE}' or '{SUPERSPREADER_STATE}'."
            )
        return get_BDSS_model(
            init_state=init_state,
            reproduction_number=skyline_parameter(self.reproduction_number, context),
            infectious_period=skyline_parameter(self.infectious_period, context),
            superspreading_ratio=skyline_parameter(self.superspreading_ratio, context),
            superspreaders_proportion=skyline_parameter(
                self.superspreaders_proportion, context
            ),
            sampling_proportion=skyline_parameter(self.sampling_proportion, context),
        )


class SIRGenerator(PhylogenieTreeGenerator):
    parameterization: Literal[Parameterization.SIR] = Parameterization.SIR
    transmission_rate: cfg.SkylineParameter
    recovery_rate: cfg.SkylineParameter
    sampling_rate: cfg.SkylineParameter
    susceptibles: cfg.Integer

    def _get_model(self, context: dict[str, Any]) -> Model:
        return get_SIR_model(
            transmission_rate=skyline_parameter(self.transmission_rate, context),
            recovery_rate=skyline_parameter(self.recovery_rate, context),
            sampling_rate=skyline_parameter(self.sampling_rate, context),
            susceptibles=integer(self.susceptibles, context),
        )


PhylogenieGeneratorConfig = Annotated[
    CanonicalGenerator
    | EpidemiologicalGenerator
    | FBDGenerator
    | BDGenerator
    | BDEIGenerator
    | BDSSGenerator
    | SIRGenerator,
    Field(discriminator="parameterization"),
]
