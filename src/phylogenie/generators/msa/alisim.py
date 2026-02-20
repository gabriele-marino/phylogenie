import subprocess
from typing import Any, Literal

from phylogenie.generators._factories import string
from phylogenie.generators.msa.base import Backend, MSAGenerator


class AliSimGenerator(MSAGenerator):
    backend: Literal[Backend.ALISIM] = Backend.ALISIM
    iqtree_path: str = "iqtree2"
    args: dict[str, Any]

    def generate(
        self,
        filename: str,
        input_tree_file: str,
        context: dict[str, Any],
        seed: int | None = None,
    ) -> None:
        command = [
            self.iqtree_path,
            "--alisim",
            filename,
            "--tree",
            input_tree_file,
            "--seed",
            str(seed),
        ]

        for key, value in self.args.items():
            command.extend([key, string(value, context)])

        command.extend(["-af", "fasta"])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["rm", f"{input_tree_file}.log"], check=True)
