from typing import Any

from numpy.random import Generator

import phylogenie.generators.configs as cfg
import phylogenie.generators.factories as f
import phylogenie.treesimulator.parameterizations.open_population as pmt
from phylogenie.generators.tree import TreeGeneratorRegistry
from phylogenie.plugins.native.base import PhylogenieTreeGenerator
from phylogenie.treesimulator import Model


@TreeGeneratorRegistry.register("phylogenie.canonical")
class CanonicalGenerator(PhylogenieTreeGenerator):
    states: list[str]
    init_state: str
    sampling_rates: cfg.SkylineVector = 0
    remove_after_sampling: bool = False
    birth_rates: cfg.SkylineVector = 0
    death_rates: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    birth_rates_among_states: cfg.SkylineMatrix = None

    def _get_model(self, context: dict[str, Any], rng: Generator) -> Model:
        return pmt.get_canonical_model(
            init_state=self.init_state.format(**context),
            states=self.states,
            sampling_rates=f.skyline_vector(self.sampling_rates, context),
            remove_after_sampling=self.remove_after_sampling,
            birth_rates=f.skyline_vector(self.birth_rates, context),
            death_rates=f.skyline_vector(self.death_rates, context),
            migration_rates=f.skyline_matrix(self.migration_rates, context),
            birth_rates_among_states=f.skyline_matrix(
                self.birth_rates_among_states, context
            ),
        )


@TreeGeneratorRegistry.register("phylogenie.FBD")
class FBDGenerator(PhylogenieTreeGenerator):
    states: list[str]
    init_state: str
    sampling_proportions: cfg.SkylineVector = 0
    diversification: cfg.SkylineVector = 0
    turnover: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    diversification_between_states: cfg.SkylineMatrix = None

    def _get_model(self, context: dict[str, Any], rng: Generator) -> Model:
        return pmt.get_FBD_model(
            init_state=self.init_state.format(**context),
            states=self.states,
            diversification=f.skyline_vector(self.diversification, context),
            turnover=f.skyline_vector(self.turnover, context),
            sampling_proportions=f.skyline_vector(self.sampling_proportions, context),
            migration_rates=f.skyline_matrix(self.migration_rates, context),
            diversification_between_states=f.skyline_matrix(
                self.diversification_between_states, context
            ),
        )


@TreeGeneratorRegistry.register("phylogenie.epidemiological")
class EpidemiologicalGenerator(PhylogenieTreeGenerator):
    states: list[str]
    init_state: str
    sampling_proportions: cfg.SkylineVector
    reproduction_numbers: cfg.SkylineVector = 0
    become_uninfectious_rates: cfg.SkylineVector = 0
    migration_rates: cfg.SkylineMatrix = None
    reproduction_numbers_among_states: cfg.SkylineMatrix = None

    def _get_model(self, context: dict[str, Any], rng: Generator) -> Model:
        return pmt.get_epidemiological_model(
            init_state=self.init_state.format(**context),
            states=self.states,
            reproduction_numbers=f.skyline_vector(self.reproduction_numbers, context),
            become_uninfectious_rates=f.skyline_vector(
                self.become_uninfectious_rates, context
            ),
            sampling_proportions=f.skyline_vector(self.sampling_proportions, context),
            migration_rates=f.skyline_matrix(self.migration_rates, context),
            reproduction_numbers_among_states=f.skyline_matrix(
                self.reproduction_numbers_among_states, context
            ),
        )


@TreeGeneratorRegistry.register("phylogenie.BD")
class BDGenerator(PhylogenieTreeGenerator):
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter

    def _get_model(self, context: dict[str, Any], rng: Generator) -> Model:
        return pmt.get_BD_model(
            reproduction_number=f.skyline_parameter(self.reproduction_number, context),
            infectious_period=f.skyline_parameter(self.infectious_period, context),
            sampling_proportion=f.skyline_parameter(self.sampling_proportion, context),
        )


@TreeGeneratorRegistry.register("phylogenie.BDEI")
class BDEIGenerator(PhylogenieTreeGenerator):
    init_state: str = pmt.EXPOSED_STATE
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    incubation_period: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter

    def _get_model(self, context: dict[str, Any], rng: Generator) -> Model:
        init_state = self.init_state.format(**context)
        if init_state not in [pmt.EXPOSED_STATE, pmt.INFECTIOUS_STATE]:
            raise ValueError(
                f"Invalid init_state '{init_state}' for BDEI model. "
                f"It must be either '{pmt.EXPOSED_STATE}' or '{pmt.INFECTIOUS_STATE}'."
            )
        return pmt.get_BDEI_model(
            init_state=init_state,
            reproduction_number=f.skyline_parameter(self.reproduction_number, context),
            infectious_period=f.skyline_parameter(self.infectious_period, context),
            incubation_period=f.skyline_parameter(self.incubation_period, context),
            sampling_proportion=f.skyline_parameter(self.sampling_proportion, context),
        )


@TreeGeneratorRegistry.register("phylogenie.BDSS")
class BDSSGenerator(PhylogenieTreeGenerator):
    init_state: str = pmt.INFECTIOUS_STATE
    reproduction_number: cfg.SkylineParameter
    infectious_period: cfg.SkylineParameter
    superspreading_ratio: cfg.SkylineParameter
    superspreaders_proportion: cfg.SkylineParameter
    sampling_proportion: cfg.SkylineParameter

    def _get_model(self, context: dict[str, Any], rng: Generator) -> Model:
        init_state = self.init_state.format(**context)
        if init_state not in [pmt.INFECTIOUS_STATE, pmt.SUPERSPREADER_STATE]:
            raise ValueError(
                f"Invalid init_state '{init_state}' for BDSS model. "
                f"It must be either '{pmt.INFECTIOUS_STATE}' or '{pmt.SUPERSPREADER_STATE}'."
            )
        return pmt.get_BDSS_model(
            init_state=init_state,
            reproduction_number=f.skyline_parameter(self.reproduction_number, context),
            infectious_period=f.skyline_parameter(self.infectious_period, context),
            superspreading_ratio=f.skyline_parameter(
                self.superspreading_ratio, context
            ),
            superspreaders_proportion=f.skyline_parameter(
                self.superspreaders_proportion, context
            ),
            sampling_proportion=f.skyline_parameter(self.sampling_proportion, context),
        )
