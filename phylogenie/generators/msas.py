import os
import subprocess
from enum import Enum
from typing import Annotated, Literal

import joblib
import pandas as pd
from pydantic import Field
from tqdm import tqdm

from phylogenie.generators.base import BaseGenerator, GeneratorType
from phylogenie.generators.trees import TreesGenerator
from phylogenie.parameterizations import Parameterization
from phylogenie.rand import RandomVariable, sample_variable

MSAS_DIRNAME = "msas"
METADATA_FILENAME = "metadata.csv"


class MSAsGeneratorBackendType(str, Enum):
    ALISIM = "alisim"


class BaseMSAsGenerator(BaseGenerator):
    type: Literal[GeneratorType.MSAS] = GeneratorType.MSAS
    backend: MSAsGeneratorBackendType


class AliSimGenerator(BaseMSAsGenerator):
    backend: Literal[MSAsGeneratorBackendType.ALISIM] = MSAsGeneratorBackendType.ALISIM
    model: str
    length: int
    branch_scale: RandomVariable = 1.0
    trees: TreesGenerator

    def generate_one(
        self,
        parameterization: Parameterization,
        branch_scale: float,
        output_file: str,
    ) -> None:
        temp_tree_file = f"{output_file}-temp.nwk"
        self.trees.generate_one(
            parameterization=parameterization,
            output_file=temp_tree_file,
        )
        command = [
            "iqtree2",
            "--alisim",
            output_file,
            "--tree",
            temp_tree_file,
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
        subprocess.run(["rm", f"{temp_tree_file}.log"], check=True)
        subprocess.run(["rm", temp_tree_file], check=True)

    def generate(
        self,
        n_samples: int,
        output_dir: str,
        n_jobs: int = -1,
    ) -> None:
        if os.path.exists(output_dir):
            print(f"Output directory {output_dir} already exists. Skipping.")
            return
        msas_dir = os.path.join(output_dir, MSAS_DIRNAME)
        os.makedirs(msas_dir, exist_ok=True)

        parameterizations = [
            self.trees.parameterization.sample() for _ in range(n_samples)
        ]
        branch_scales = [sample_variable(self.branch_scale) for _ in range(n_samples)]

        iterator = tqdm(
            enumerate(zip(parameterizations, branch_scales)),
            total=n_samples,
            desc=f"Generating MSAs ({output_dir})",
        )
        if n_jobs == 1:
            for i, (parameterization, branch_scale) in iterator:
                self.generate_one(
                    parameterization=parameterization,
                    branch_scale=branch_scale,
                    output_file=os.path.join(msas_dir, str(i)),
                )
        else:
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.generate_one)(
                    parameterization=parameterization,
                    branch_scale=branch_scale,
                    output_file=os.path.join(msas_dir, str(i)),
                )
                for i, (parameterization, branch_scale) in iterator
            )

        df = pd.DataFrame([p.serialize() for p in parameterizations])
        df["branch_scale"] = branch_scales
        df.insert(0, "filename", tuple(f"{i}.fa" for i in range(n_samples)))
        df.to_csv(os.path.join(output_dir, METADATA_FILENAME), index=False)


MSAsGenerator = Annotated[AliSimGenerator, Field(discriminator="backend")]
