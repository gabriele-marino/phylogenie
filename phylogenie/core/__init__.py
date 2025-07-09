from typing import Annotated

from pydantic import Field

from phylogenie.core.dataset import DatasetGenerator
from phylogenie.core.msas import MSAsGeneratorConfig
from phylogenie.core.trees import TreesGeneratorConfig

DatasetGeneratorConfig = Annotated[
    TreesGeneratorConfig | MSAsGeneratorConfig,
    Field(discriminator="data_type"),
]

__all__ = ["DatasetGeneratorConfig", "DatasetGenerator"]
