import os
import subprocess
from pathlib import Path
from typing import Any, Literal

from numpy.random import Generator

from phylogenie.generators.dataset import DatasetGenerator, DataType
from phylogenie.generators.trees import TreeDatasetGeneratorConfig
from phylogenie.io import dump_newick

MSAS_DIRNAME = "MSAs"
TREES_DIRNAME = "trees"


class AliSimDatasetGenerator(DatasetGenerator):
    data_type: Literal[DataType.MSAS] = DataType.MSAS
    trees: TreeDatasetGeneratorConfig
    keep_trees: bool = False
    iqtree_path: str = "iqtree2"
    args: dict[str, str | int | float]

    def _generate_one_from_tree(
        self, filename: str, tree_file: str, rng: Generator, data: dict[str, Any]
    ) -> None:
        command = [
            self.iqtree_path,
            "--alisim",
            filename,
            "--tree",
            tree_file,
            "--seed",
            str(rng.integers(2**32)),
        ]

        for key, value in self.args.items():
            command.extend(
                [key, value.format(**data) if isinstance(value, str) else str(value)]
            )

        command.extend(["-af", "fasta"])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["rm", f"{tree_file}.log"], check=True)

    def _generate_one(
        self, filename: str, rng: Generator, data: dict[str, Any]
    ) -> None:
        if self.keep_trees:
            base_dir, file_id = Path(filename).parent, Path(filename).stem
            trees_dir = os.path.join(base_dir, TREES_DIRNAME)
            msas_dir = os.path.join(base_dir, MSAS_DIRNAME)
            os.makedirs(trees_dir, exist_ok=True)
            os.makedirs(msas_dir, exist_ok=True)
            tree_filename = os.path.join(trees_dir, file_id)
            msa_filename = os.path.join(msas_dir, file_id)
        else:
            tree_filename = f"{filename}.temp-tree"
            msa_filename = filename

        tree = self.trees.simulate_one(rng, data)
        if tree is None:
            return

        for leaf in tree.get_leaves():
            leaf.id += f"|{leaf.get_time()}"
        dump_newick(tree, f"{tree_filename}.nwk")

        self._generate_one_from_tree(
            filename=msa_filename, tree_file=f"{tree_filename}.nwk", rng=rng, data=data
        )
        if not self.keep_trees:
            os.remove(f"{tree_filename}.nwk")
