from typing import Annotated

from pydantic import Field

from phylogenie.core.trees.remaster import ReMASTERGeneratorConfig
from phylogenie.core.trees.treesimulator import TreeSimulatorGeneratorConfig

TreesGeneratorConfig = Annotated[
    ReMASTERGeneratorConfig | TreeSimulatorGeneratorConfig,
    Field(discriminator="backend"),
]
