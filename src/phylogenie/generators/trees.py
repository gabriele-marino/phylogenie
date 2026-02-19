from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from numpy.random import default_rng
from pydantic import Field

import phylogenie.generators._configs as cfg
from phylogenie.generators._factories import (
    data,
    eval_expression,
    integer,
    scalar,
    skyline_matrix,
    skyline_parameter,
    skyline_vector,
    timed_event,
)
from phylogenie.generators.dataset import DatasetGenerator, DataType
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


class ParameterizationType(str, Enum):
    CANONICAL = "canonical"
    EPIDEMIOLOGICAL = "epidemiological"
    FBD = "FBD"
    BD = "BD"
    BDEI = "BDEI"
    BDSS = "BDSS"
    SIR = "SIR"


class TreeDatasetGenerator(DatasetGenerator):
    data_type: Literal[DataType.TREES] = DataType.TREES
    n_leaves: cfg.Integer | None = None
    max_time: cfg.Scalar | None = None
    timeout: float | None = None
    timed_events: Annotated[list[cfg.TimedEvent], Field(default_factory=list)]
    node_features: dict[str, str] | None = None
    acceptance_criterion: str | None = None
    logs: dict[str, str] | None = None

    @abstractmethod
    def _get_model(self, data: dict[str, Any]) -> Model: ...

    def simulate_one(
        self, data: dict[str, Any], seed: int | None = None
    ) -> tuple[TreeNode, dict[str, Any]]:
        model = self._get_model(data)
        model.rng.seed(seed)

        for event in self.timed_events:
            model.add_timed_event(timed_event(event, data))

        max_time = None if self.max_time is None else scalar(self.max_time, data)

        acceptance_criterion: None | Callable[[TreeNode], bool] = (
            None
            if self.acceptance_criterion is None
            else lambda tree: eval_expression(
                self.acceptance_criterion,  # pyright: ignore
                data,
                {"tree": tree},
            )
        )
        logs: None | dict[str, Callable[[TreeNode], Any]] = (
            None
            if self.logs is None
            else {
                key: lambda tree, expr=expr: eval_expression(expr, data, {"tree": tree})
                for key, expr in self.logs.items()
            }
        )

        return simulate_tree(
            model=model,
            n_leaves=None if self.n_leaves is None else integer(self.n_leaves, data),
            max_time=max_time,
            timeout=self.timeout,
            acceptance_criterion=acceptance_criterion,
            logs=logs,
        )

    def generate_one(self, filename: str, seed: int | None = None) -> dict[str, Any]:
        d = {"file_id": Path(filename).stem}
        rng = default_rng(seed)
        while True:
            try:
                d.update(data(self.context, rng))
                tree, metadata = self.simulate_one(d, seed)
                if self.node_features is not None:
                    for name, feature in self.node_features.items():
                        mapping = getattr(tree, feature)
                        for node in tree:
                            node[name] = mapping[node]
                dump_newick(tree, f"{filename}.nwk")
                break
            except TimeoutError as e:
                print(f"{e}. Retrying with different parameters...")
        return d | metadata


class CanonicalTreeDatasetGenerator(TreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.CANONICAL] = (
        ParameterizationType.CANONICAL
    )
    states: list[str]
    init_state: str
    sampling_rates: cfg.SkylineVector = 0
    remove_after_sampling: bool = False
    birth_rates: cfg.SkylineVector = 0
    death_rates: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    birth_rates_among_states: cfg.SkylineMatrix = None

    def _get_model(self, data: dict[str, Any]) -> Model:
        return get_canonical_model(
            init_state=self.init_state.format(**data),
            states=self.states,
            sampling_rates=skyline_vector(self.sampling_rates, data),
            remove_after_sampling=self.remove_after_sampling,
            birth_rates=skyline_vector(self.birth_rates, data),
            death_rates=skyline_vector(self.death_rates, data),
            migration_rates=skyline_matrix(self.migration_rates, data),
            birth_rates_among_states=skyline_matrix(
                self.birth_rates_among_states, data
            ),
        )


class FBDTreeDatasetGenerator(TreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.FBD] = ParameterizationType.FBD
    states: list[str]
    init_state: str
    sampling_proportions: cfg.SkylineVector = 0
    diversification: cfg.SkylineVector = 0
    turnover: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    diversification_between_states: cfg.SkylineMatrix = None

    def _get_model(self, data: dict[str, Any]) -> Model:
        return get_FBD_model(
            init_state=self.init_state.format(**data),
            states=self.states,
            diversification=skyline_vector(self.diversification, data),
            turnover=skyline_vector(self.turnover, data),
            sampling_proportions=skyline_vector(self.sampling_proportions, data),
            migration_rates=skyline_matrix(self.migration_rates, data),
            diversification_between_states=skyline_matrix(
                self.diversification_between_states, data
            ),
        )


class EpidemiologicalTreeDatasetGenerator(TreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.EPIDEMIOLOGICAL] = (
        ParameterizationType.EPIDEMIOLOGICAL
    )
    states: list[str]
    init_state: str
    sampling_proportions: cfg.SkylineVector
    reproduction_numbers: cfg.SkylineVector = 0
    become_uninfectious_rates: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    reproduction_numbers_among_states: cfg.SkylineMatrix = None

    def _get_model(self, data: dict[str, Any]) -> Model:
        return get_epidemiological_model(
            init_state=self.init_state.format(**data),
            states=self.states,
            reproduction_numbers=skyline_vector(self.reproduction_numbers, data),
            become_uninfectious_rates=skyline_vector(
                self.become_uninfectious_rates, data
            ),
            sampling_proportions=skyline_vector(self.sampling_proportions, data),
            migration_rates=skyline_matrix(self.migration_rates, data),
            reproduction_numbers_among_states=skyline_matrix(
                self.reproduction_numbers_among_states, data
            ),
        )


class BDTreeDatasetGenerator(TreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.BD] = ParameterizationType.BD
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter

    def _get_model(self, data: dict[str, Any]) -> Model:
        return get_BD_model(
            reproduction_number=skyline_parameter(self.reproduction_number, data),
            infectious_period=skyline_parameter(self.infectious_period, data),
            sampling_proportion=skyline_parameter(self.sampling_proportion, data),
        )


class BDEITreeDatasetGenerator(TreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.BDEI] = ParameterizationType.BDEI
    init_state: str = EXPOSED_STATE
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    incubation_period: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter

    def _get_model(self, data: dict[str, Any]) -> Model:
        init_state = self.init_state.format(**data)
        if init_state not in [EXPOSED_STATE, INFECTIOUS_STATE]:
            raise ValueError(
                f"Invalid init_state '{init_state}' for BDEI model. "
                f"It must be either '{EXPOSED_STATE}' or '{INFECTIOUS_STATE}'."
            )
        return get_BDEI_model(
            init_state=init_state,
            reproduction_number=skyline_parameter(self.reproduction_number, data),
            infectious_period=skyline_parameter(self.infectious_period, data),
            incubation_period=skyline_parameter(self.incubation_period, data),
            sampling_proportion=skyline_parameter(self.sampling_proportion, data),
        )


class BDSSTreeDatasetGenerator(TreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.BDSS] = ParameterizationType.BDSS
    init_state: str = INFECTIOUS_STATE
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    superspreading_ratio: cfg.SkylineParameter
    superspreaders_proportion: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter

    def _get_model(self, data: dict[str, Any]) -> Model:
        init_state = self.init_state.format(**data)
        if init_state not in [INFECTIOUS_STATE, SUPERSPREADER_STATE]:
            raise ValueError(
                f"Invalid init_state '{init_state}' for BDSS model. "
                f"It must be either '{INFECTIOUS_STATE}' or '{SUPERSPREADER_STATE}'."
            )
        return get_BDSS_model(
            init_state=init_state,
            reproduction_number=skyline_parameter(self.reproduction_number, data),
            infectious_period=skyline_parameter(self.infectious_period, data),
            superspreading_ratio=skyline_parameter(self.superspreading_ratio, data),
            superspreaders_proportion=skyline_parameter(
                self.superspreaders_proportion, data
            ),
            sampling_proportion=skyline_parameter(self.sampling_proportion, data),
        )


class SIRTreeDatasetGenerator(TreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.SIR] = ParameterizationType.SIR
    transmission_rate: cfg.SkylineParameter
    recovery_rate: cfg.SkylineParameter
    sampling_rate: cfg.SkylineParameter
    susceptibles: cfg.Integer

    def _get_model(self, data: dict[str, Any]) -> Model:
        return get_SIR_model(
            transmission_rate=skyline_parameter(self.transmission_rate, data),
            recovery_rate=skyline_parameter(self.recovery_rate, data),
            sampling_rate=skyline_parameter(self.sampling_rate, data),
            susceptibles=integer(self.susceptibles, data),
        )


TreeDatasetGeneratorConfig = Annotated[
    CanonicalTreeDatasetGenerator
    | EpidemiologicalTreeDatasetGenerator
    | FBDTreeDatasetGenerator
    | BDTreeDatasetGenerator
    | BDEITreeDatasetGenerator
    | BDSSTreeDatasetGenerator
    | SIRTreeDatasetGenerator,
    Field(discriminator="parameterization"),
]
