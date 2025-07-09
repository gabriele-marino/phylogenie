import subprocess
from typing import Literal

from numpy.random import Generator

from phylogenie.core.msas.base import BackendType, MSAsGenerator
from phylogenie.core.typings import Data


class AliSimGenerator(MSAsGenerator):
    backend: Literal[BackendType.ALISIM] = BackendType.ALISIM
    iqtree_path: str = "iqtree2"
    args: dict[str, str | int | float]

    def _generate_one_from_tree(
        self, filename: str, tree_file: str, rng: Generator, data: Data
    ) -> None:
        command = [
            self.iqtree_path,
            "--alisim",
            filename,
            "--tree",
            tree_file,
            "--seed",
            str(rng.integers(0, 2**32 - 1)),
        ]

        for key, value in self.args.items():
            command.extend(
                [key, value.format(**data) if isinstance(value, str) else str(value)]
            )

        command.extend(["-af", "fasta"])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["rm", f"{tree_file}.log"], check=True)
