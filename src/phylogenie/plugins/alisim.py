import subprocess
from typing import Any

import phylogenie.generators.factories as f
from phylogenie.generators.msa import MSAGenerator, MSAGeneratorRegistry


@MSAGeneratorRegistry.register("alisim")
class AliSimGenerator(MSAGenerator):
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
            command.extend([key, f.string(value, context)])

        command.extend(["-af", "fasta"])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["rm", f"{input_tree_file}.log"], check=True)
