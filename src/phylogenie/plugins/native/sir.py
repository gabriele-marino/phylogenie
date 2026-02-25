from typing import Any

from numpy.random import Generator

import phylogenie.generators.configs as cfg
import phylogenie.generators.factories as f
import phylogenie.treesimulator.parameterizations.sir as pmt
from phylogenie.generators.tree import TreeGeneratorRegistry
from phylogenie.plugins.native.base import PhylogenieTreeGenerator
from phylogenie.treesimulator import Model


@TreeGeneratorRegistry.register("phylogenie.SIR")
class SIRGenerator(PhylogenieTreeGenerator):
    transmission_rate: cfg.SkylineParameter
    recovery_rate: cfg.SkylineParameter
    sampling_rate: cfg.SkylineParameter
    susceptibles: cfg.Integer

    def _get_model(self, context: dict[str, Any], rng: Generator) -> Model:
        return pmt.get_SIR_model(
            transmission_rate=f.skyline_parameter(self.transmission_rate, context),
            recovery_rate=f.skyline_parameter(self.recovery_rate, context),
            sampling_rate=f.skyline_parameter(self.sampling_rate, context),
            susceptibles=f.integer(self.susceptibles, context),
        )
