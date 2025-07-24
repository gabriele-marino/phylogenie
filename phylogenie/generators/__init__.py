from typing import Annotated

from pydantic import Field

from phylogenie.generators.dataset import DatasetGenerator
from phylogenie.generators.msas import MSAsGeneratorConfig
from phylogenie.generators.trees import TreesGeneratorConfig

DatasetGeneratorConfig = Annotated[
    TreesGeneratorConfig | MSAsGeneratorConfig,
    Field(discriminator="data_type"),
]

__all__ = ["DatasetGeneratorConfig", "DatasetGenerator"]
