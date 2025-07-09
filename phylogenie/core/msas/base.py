import os
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Literal

from numpy.random import Generator

from phylogenie.core.dataset import DatasetGenerator, DataType
from phylogenie.core.trees import TreesGeneratorConfig
from phylogenie.core.typings import Data


class BackendType(str, Enum):
    ALISIM = "alisim"


class MSAsGenerator(DatasetGenerator):
    data_type: Literal[DataType.MSAS] = DataType.MSAS
    trees: str | TreesGeneratorConfig
    output_trees_dir: str | None = None

    @abstractmethod
    def _generate_one_from_tree(
        self, filename: str, tree_file: str, rng: Generator, data: Data
    ) -> None: ...

    def _generate_one(self, filename: str, rng: Generator, data: Data) -> None:
        if isinstance(self.trees, str):
            tree_files = os.listdir(self.trees)
            tree_file = os.path.join(self.trees, str(rng.choice(tree_files)))
            self._generate_one_from_tree(
                filename=filename, tree_file=tree_file, rng=rng, data=data
            )
        elif isinstance(self.output_trees_dir, str):
            os.makedirs(self.output_trees_dir, exist_ok=True)
            self._generate_one_from_tree(
                filename=filename,
                tree_file=os.path.join(self.output_trees_dir, f"{Path(filename).stem}"),
                rng=rng,
                data=data,
            )
        else:
            tree_filename = f"{filename}.temp-tree"
            self.trees.generate_one(
                filename=tree_filename, data=data, seed=int(rng.integers(0, 2**32 - 1))
            )
            self._generate_one_from_tree(
                filename=filename, tree_file=f"{tree_filename}.nwk", rng=rng, data=data
            )
            os.remove(f"{tree_filename}.nwk")
