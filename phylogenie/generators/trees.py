import os
from abc import abstractmethod
from enum import Enum
from typing import Annotated, Any, Literal

import joblib
import pandas as pd
from pydantic import Field
from pykit.type_hints import OneOrSequence
from tqdm import tqdm

from phylogenie.generators.base import BaseGenerator, GeneratorType
from phylogenie.helpers import remaster, treesimulator
from phylogenie.helpers.remaster import PunctualReaction
from phylogenie.parameterizations import Parameterization, RandomParameterization

TREES_DIRNAME = "trees"
METADATA_FILENAME = "metadata.csv"


class TreesGeneratorBackendType(str, Enum):
    REMASTER = "remaster"
    TREESIMULATOR = "treesimulator"


class BaseTreesGenerator(BaseGenerator):
    type: Literal[GeneratorType.TREES] = GeneratorType.TREES
    backend: TreesGeneratorBackendType
    parameterization: RandomParameterization

    @abstractmethod
    def generate_one(
        self,
        parameterization: Parameterization,
        output_file: str,
    ) -> None: ...

    def generate(
        self,
        n_samples: int,
        output_dir: str,
        n_jobs: int = -1,
    ) -> None:
        if os.path.exists(output_dir):
            print(f"Output directory {output_dir} already exists. Skipping.")
            return
        trees_dir = os.path.join(output_dir, TREES_DIRNAME)
        os.makedirs(trees_dir, exist_ok=True)

        parameterizations = [self.parameterization.sample() for _ in range(n_samples)]
        iterator = tqdm(
            enumerate(parameterizations),
            total=n_samples,
            desc=f"Generating trees ({output_dir})",
        )
        if n_jobs == 1:
            for i, parameterization in iterator:
                self.generate_one(
                    parameterization=parameterization,
                    output_file=os.path.join(trees_dir, f"{i}.nwk"),
                )
        else:
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.generate_one)(
                    parameterization=parameterization,
                    output_file=os.path.join(trees_dir, f"{i}.nwk"),
                )
                for i, parameterization in iterator
            )

        df = pd.DataFrame([p.serialize() for p in parameterizations])
        df.insert(0, "filename", tuple(f"{i}.nwk" for i in range(n_samples)))
        df.to_csv(os.path.join(output_dir, METADATA_FILENAME), index=False)


class ReMASTERGenerator(BaseTreesGenerator):
    backend: Literal[TreesGeneratorBackendType.REMASTER] = (
        TreesGeneratorBackendType.REMASTER
    )
    init_values: list[int] | None = None
    punctual_reactions: OneOrSequence[PunctualReaction] | None = None
    trajectory_attrs: dict[str, str] | None = None
    remove_singleton_nodes: bool = True

    def generate_one(
        self,
        parameterization: Parameterization,
        output_file: str,
    ) -> None:
        remaster.generate_trees(
            parameterization=parameterization,
            init_values=(
                [1] + [0] * (len(parameterization.populations) - 1)
                if self.init_values is None
                else self.init_values
            ),
            output_file=output_file,
            punctual_reactions=self.punctual_reactions,
            trajectory_attrs=self.trajectory_attrs,
            remove_singleton_nodes=self.remove_singleton_nodes,
        )


class TreesimulatorGenerator(BaseTreesGenerator):
    backend: Literal[TreesGeneratorBackendType.TREESIMULATOR] = (
        TreesGeneratorBackendType.TREESIMULATOR
    )
    kwargs: dict[str, Any] = Field(default_factory=dict)

    def generate_one(
        self,
        parameterization: Parameterization,
        output_file: str,
    ) -> None:
        treesimulator.generate_tree(
            parameterization=parameterization,
            output_file=output_file,
            **self.kwargs,
        )


TreesGenerator = Annotated[
    ReMASTERGenerator | TreesimulatorGenerator, Field(discriminator="backend")
]
