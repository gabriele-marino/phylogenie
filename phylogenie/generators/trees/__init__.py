from typing import Annotated

from pydantic import Field

from phylogenie.generators.trees.remaster import ReMASTERGeneratorConfig
from phylogenie.generators.trees.treesimulator import TreeSimulatorGeneratorConfig

TreesGeneratorConfig = Annotated[
    ReMASTERGeneratorConfig | TreeSimulatorGeneratorConfig,
    Field(discriminator="backend"),
]
