import os
from abc import ABC, abstractmethod
from typing import Annotated, Any

import phylogenie.generators.configs as cfg
from phylogenie.generators.dataset import DATASET_GENERATOR_REGISTRY, DatasetGenerator
from phylogenie.utils import Registry


class TreeGenerator(ABC, cfg.ForbidExtraBaseModel):
    @abstractmethod
    def generate(
        self, filename: str, context: dict[str, Any], seed: int | None = None
    ) -> dict[str, Any]: ...


TREE_GENERATOR_REGISTRY: Registry[TreeGenerator] = Registry(TreeGenerator)
TreeGeneratorConfig = Annotated[TreeGenerator, TREE_GENERATOR_REGISTRY.validator]


@DATASET_GENERATOR_REGISTRY.register("tree")
class TreeDatasetGenerator(DatasetGenerator):
    tree_generator: TreeGeneratorConfig

    def _generate_from_context(
        self, data_dir: str, file_id: str, context: dict[str, Any], seed: int
    ) -> dict[str, Any]:
        return self.tree_generator.generate(
            os.path.join(data_dir, file_id), context, seed
        )
