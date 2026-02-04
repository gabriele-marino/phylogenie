from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from numpy.random import default_rng
from pydantic import Field

import phylogenie.generators._configs as cfg
from phylogenie.generators._factories import (
    add_unbounded_population_timed_event,
    data,
    eval_expression,
    integer,
    scalar,
    skyline_matrix,
    skyline_parameter,
    skyline_vector,
)
from phylogenie.generators.dataset import DatasetGenerator, DataType
from phylogenie.io import dump_newick
from phylogenie.tree_node import TreeNode
from phylogenie.treesimulator import (
    EXPOSED_STATE,
    INFECTIOUS_STATE,
    SUPERSPREADER_STATE,
    Model,
    UnboundedPopulationModel,
    get_BD_model,
    get_BDEI_model,
    get_BDSS_model,
    get_canonical_model,
    get_epidemiological_model,
    get_FBD_model,
    simulate_tree,
)


class ParameterizationType(str, Enum):
    CANONICAL = "canonical"
    EPIDEMIOLOGICAL = "epidemiological"
    FBD = "FBD"
    BD = "BD"
    BDEI = "BDEI"
    BDSS = "BDSS"


class TreeDatasetGenerator(DatasetGenerator):
    data_type: Literal[DataType.TREES] = DataType.TREES
    n_leaves: cfg.Integer | None = None
    max_time: cfg.Scalar | None = None
    timeout: float | None = None
    node_features: dict[str, str] | None = None
    acceptance_criterion: str | None = None
    logs: dict[str, str] | None = None

    @abstractmethod
    def _get_model(self, data: dict[str, Any], seed: int | None) -> Model: ...

    def simulate_one(
        self, data: dict[str, Any], seed: int | None = None
    ) -> tuple[TreeNode, dict[str, Any]]:
        model = self._get_model(data, seed)

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
            timeout=self.timeout,
            acceptance_criterion=acceptance_criterion,
            logs=logs,
        )

    def generate_one(
        self,
        filename: str,
        context: dict[str, cfg.Distribution] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        d = {"file_id": Path(filename).stem}
        rng = default_rng(seed)
        while True:
            try:
                d.update(data(context, rng))
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


class UnboundedPopulationTreeDatasetGenerator(TreeDatasetGenerator):
    timed_events: Annotated[
        list[cfg.UnboundedPopulationTimedEvent], Field(default_factory=list)
    ]

    @abstractmethod
    def _get_base_model(
        self, data: dict[str, Any], seed: int | None
    ) -> UnboundedPopulationModel: ...

    def _get_model(
        self, data: dict[str, Any], seed: int | None
    ) -> UnboundedPopulationModel:
        model = self._get_base_model(data, seed)
        for timed_event in self.timed_events:
            add_unbounded_population_timed_event(model, timed_event, data)
        return model


class CanonicalTreeDatasetGenerator(UnboundedPopulationTreeDatasetGenerator):
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

    def _get_base_model(
        self, data: dict[str, Any], seed: int | None
    ) -> UnboundedPopulationModel:
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
            max_time=None if self.max_time is None else scalar(self.max_time, data),
            seed=seed,
        )


class FBDTreeDatasetGenerator(UnboundedPopulationTreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.FBD] = ParameterizationType.FBD
    states: list[str]
    init_state: str
    sampling_proportions: cfg.SkylineVector = 0
    diversification: cfg.SkylineVector = 0
    turnover: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    diversification_between_states: cfg.SkylineMatrix = None

    def _get_base_model(
        self, data: dict[str, Any], seed: int | None
    ) -> UnboundedPopulationModel:
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
            max_time=None if self.max_time is None else scalar(self.max_time, data),
            seed=seed,
        )


class EpidemiologicalTreeDatasetGenerator(UnboundedPopulationTreeDatasetGenerator):
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

    def _get_base_model(
        self, data: dict[str, Any], seed: int | None
    ) -> UnboundedPopulationModel:
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
            max_time=None if self.max_time is None else scalar(self.max_time, data),
            seed=seed,
        )


class BDTreeDatasetGenerator(UnboundedPopulationTreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.BD] = ParameterizationType.BD
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter

    def _get_base_model(
        self, data: dict[str, Any], seed: int | None
    ) -> UnboundedPopulationModel:
        return get_BD_model(
            reproduction_number=skyline_parameter(self.reproduction_number, data),
            infectious_period=skyline_parameter(self.infectious_period, data),
            sampling_proportion=skyline_parameter(self.sampling_proportion, data),
            max_time=None if self.max_time is None else scalar(self.max_time, data),
            seed=seed,
        )


class BDEITreeDatasetGenerator(UnboundedPopulationTreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.BDEI] = ParameterizationType.BDEI
    init_state: str = EXPOSED_STATE
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    incubation_period: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter

    def _get_base_model(
        self, data: dict[str, Any], seed: int | None
    ) -> UnboundedPopulationModel:
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
            max_time=None if self.max_time is None else scalar(self.max_time, data),
            seed=seed,
        )


class BDSSTreeDatasetGenerator(UnboundedPopulationTreeDatasetGenerator):
    parameterization: Literal[ParameterizationType.BDSS] = ParameterizationType.BDSS
    init_state: str = INFECTIOUS_STATE
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    superspreading_ratio: cfg.SkylineParameter
    superspreaders_proportion: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter

    def _get_base_model(
        self, data: dict[str, Any], seed: int | None
    ) -> UnboundedPopulationModel:
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
            max_time=None if self.max_time is None else scalar(self.max_time, data),
            seed=seed,
        )


TreeDatasetGeneratorConfig = Annotated[
    CanonicalTreeDatasetGenerator
    | EpidemiologicalTreeDatasetGenerator
    | FBDTreeDatasetGenerator
    | BDTreeDatasetGenerator
    | BDEITreeDatasetGenerator
    | BDSSTreeDatasetGenerator,
    Field(discriminator="parameterization"),
]
