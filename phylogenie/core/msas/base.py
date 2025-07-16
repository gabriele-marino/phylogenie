import os
from abc import abstractmethod
from enum import Enum
from typing import Literal

from numpy.random import Generator

import phylogenie.typings as pgt
from phylogenie.core.dataset import DatasetGenerator, DataType
from phylogenie.core.trees import TreesGeneratorConfig


class BackendType(str, Enum):
    ALISIM = "alisim"


class MSAsGenerator(DatasetGenerator):
    data_type: Literal[DataType.MSAS] = DataType.MSAS
    trees: TreesGeneratorConfig

    @abstractmethod
    def _generate_one_from_tree(
        self, filename: str, tree_file: str, rng: Generator, data: pgt.Data
    ) -> None: ...

    def _generate_one(self, filename: str, rng: Generator, data: pgt.Data) -> None:
        tree_filename = f"{filename}.temp-tree"
        self.trees.generate_one(
            filename=tree_filename, data=data, seed=int(rng.integers(0, 2**32 - 1))
        )
        self._generate_one_from_tree(
            filename=filename, tree_file=f"{tree_filename}.nwk", rng=rng, data=data
        )
        os.remove(f"{tree_filename}.nwk")
