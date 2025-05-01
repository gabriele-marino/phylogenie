import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import joblib
import pandas as pd
from pydantic import Field
from tqdm import tqdm

from phylogenie.generators.base import BaseGenerator, GeneratorType
from phylogenie.generators.trees import BaseTreesGenerator, TreesGenerator
from phylogenie.rand import RandomVariable, sample_variable


class MSAsGeneratorBackendType(str, Enum):
    ALISIM = "alisim"


class BaseMSAsGenerator(BaseGenerator):
    type: Literal[GeneratorType.MSAS] = GeneratorType.MSAS
    backend: MSAsGeneratorBackendType


class AliSimGenerator(BaseMSAsGenerator):
    backend: Literal[MSAsGeneratorBackendType.ALISIM] = MSAsGeneratorBackendType.ALISIM
    model: str
    length: int
    trees: str | TreesGenerator
    branch_scale: RandomVariable = 1.0
    trees_metadata_file: str | None = None

    def generate_one(
        self,
        tree_file: str,
        output_file: str,
        branch_scale: float,
    ) -> None:
        command = [
            "iqtree2",
            "--alisim",
            output_file,
            "--tree",
            tree_file,
            "-m",
            self.model,
            "--length",
            str(self.length),
            "--branch-scale",
            str(branch_scale),
            "-af",
            "fasta",
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["rm", f"{tree_file}.log"], check=True)

    def generate(
        self,
        n_samples: int,
        output_dir: str,
        output_metadata_file: str,
        n_jobs: int = -1,
    ) -> None:
        if os.path.exists(output_dir):
            print(f"Output directory {output_dir} already exists. Skipping.")
            return
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(self.trees, str):
            trees_dir = self.trees
        else:
            trees_dir = os.path.join(output_dir, ".trees")

        if isinstance(self.trees, BaseTreesGenerator):
            output_trees_metadata_file = os.path.join(output_dir, ".trees-metadata.csv")
            self.trees.generate(
                n_samples=n_samples,
                output_dir=trees_dir,
                output_metadata_file=output_trees_metadata_file,
                n_jobs=n_jobs,
            )

        tree_files = [
            os.path.join(trees_dir, file)
            for file in os.listdir(trees_dir)
            if file.endswith(".nwk")
        ]
        ids = [Path(file).stem for file in tree_files]
        msa_files = [os.path.join(output_dir, id) for id in ids]
        branch_scales = [sample_variable(self.branch_scale) for _ in ids]

        if n_jobs == 1:
            for tree_file, msa_file, branch_scale in tqdm(
                zip(tree_files, msa_files, branch_scales),
                total=len(msa_files),
                desc=f"Generating MSAs ({output_dir})",
            ):
                self.generate_one(
                    tree_file=tree_file,
                    output_file=msa_file,
                    branch_scale=branch_scale,
                )
        else:
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.generate_one)(
                    tree_file=tree_file,
                    branch_scale=branch_scale,
                    output_file=msa_file,
                )
                for tree_file, msa_file, branch_scale in tqdm(
                    zip(tree_files, msa_files, branch_scales),
                    total=len(msa_files),
                    desc=f"Generating MSAs ({output_dir})",
                )
            )

        if output_metadata_file is not None:
            metadata = pd.DataFrame(
                {
                    "filename": tuple(f"{id}.fa" for id in ids),
                    "branch_scale": branch_scales,
                }
            )
            trees_metadata_file: str | None
            if isinstance(self.trees, BaseTreesGenerator):
                trees_metadata_file = output_trees_metadata_file
            else:
                trees_metadata_file = self.trees_metadata_file
            if trees_metadata_file is not None:
                metadata["file_id"] = tuple(str(id) for id in ids)
                trees_metadata = pd.read_csv(trees_metadata_file)
                trees_metadata["file_id"] = trees_metadata["filename"].apply(
                    lambda fn: str(Path(fn).stem)
                )
                trees_metadata = trees_metadata.drop(columns=["filename"])
                metadata = pd.merge(metadata, trees_metadata, on="file_id")
                metadata = metadata.drop(columns=["file_id"])
            metadata.to_csv(output_metadata_file, index=False)

        if isinstance(self.trees, BaseTreesGenerator):
            subprocess.run(["rm", "-rf", trees_dir], check=True)
            subprocess.run(["rm", "-rf", output_trees_metadata_file], check=True)


MSAsGenerator = Annotated[AliSimGenerator, Field(discriminator="backend")]
