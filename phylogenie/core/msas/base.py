import os
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Literal

from numpy.random import Generator

import phylogenie.typings as pgt
from phylogenie.core.dataset import DatasetGenerator, DataType
from phylogenie.core.trees import TreesGeneratorConfig


class BackendType(str, Enum):
    ALISIM = "alisim"


MSAS_DIRNAME = "MSAs"
TREES_DIRNAME = "trees"


class MSAsGenerator(DatasetGenerator):
    data_type: Literal[DataType.MSAS] = DataType.MSAS
    trees: TreesGeneratorConfig
    keep_trees: bool = False

    @abstractmethod
    def _generate_one_from_tree(
        self, filename: str, tree_file: str, rng: Generator, data: pgt.Data
    ) -> None: ...

    def _generate_one(self, filename: str, rng: Generator, data: pgt.Data) -> None:
        if self.keep_trees:
            base_dir = Path(filename).parent
            file_id = Path(filename).stem
            tree_filename = os.path.join(base_dir, TREES_DIRNAME, file_id)
            msas_dir = os.path.join(base_dir, MSAS_DIRNAME)
            os.makedirs(msas_dir, exist_ok=True)
            msa_filename = os.path.join(msas_dir, file_id)
        else:
            tree_filename = f"{filename}.temp-tree"
            msa_filename = filename

        self.trees.generate_one(
            filename=tree_filename, data=data, seed=int(rng.integers(0, 2**32 - 1))
        )
        self._generate_one_from_tree(
            filename=msa_filename, tree_file=f"{tree_filename}.nwk", rng=rng, data=data
        )
        if not self.keep_trees:
            os.remove(f"{tree_filename}.nwk")
