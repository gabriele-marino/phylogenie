import os
import subprocess
from abc import abstractmethod
from enum import Enum
from typing import Annotated, Any, Literal

import joblib
import pandas as pd
from pydantic import Field
from tqdm import tqdm

from phylogenie.generators.base import BaseGenerator, GeneratorType
from phylogenie.helpers.remaster import prepare_config_file
from phylogenie.helpers.treesimulator import generate_tree
from phylogenie.parameterizations import RandomParameterization, Rates
from phylogenie.utils import extract_newick_from_nexus, process_newick_taxa_names


class TreesGeneratorBackendType(str, Enum):
    REMASTER = "remaster"
    TREESIMULATOR = "treesimulator"


class BaseTreesGenerator(BaseGenerator):
    type: Literal[GeneratorType.TREES] = GeneratorType.TREES
    backend: TreesGeneratorBackendType
    parameterization: RandomParameterization

    @abstractmethod
    def _generate_one(
        self,
        rates: Rates,
        populations: list[str],
        init_values: list[int],
        output_file: str,
    ) -> None: ...

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

        parameterizations = [self.parameterization.sample() for _ in range(n_samples)]
        rates = [p.rates for p in parameterizations]

        if n_jobs == 1:
            for i, r in tqdm(
                enumerate(rates),
                total=n_samples,
                desc=f"Generating trees ({output_dir})",
            ):
                self._generate_one(
                    rates=r,
                    populations=self.parameterization.populations,
                    init_values=self.parameterization.init_values,
                    output_file=os.path.join(output_dir, f"{i}.nwk"),
                )
        else:
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self._generate_one)(
                    rates=r,
                    populations=self.parameterization.populations,
                    init_values=self.parameterization.init_values,
                    output_file=os.path.join(output_dir, f"{i}.nwk"),
                )
                for i, r in tqdm(
                    enumerate(rates),
                    total=n_samples,
                    desc=f"Generating trees ({output_dir})",
                )
            )

        if output_metadata_file is not None:
            df = pd.DataFrame([p.serialize() for p in parameterizations])
            df.insert(0, "filename", tuple(f"{i}.nwk" for i in range(n_samples)))
            df.to_csv(output_metadata_file, index=False)


class ReMASTERGenerator(BaseTreesGenerator):
    backend: Literal[TreesGeneratorBackendType.REMASTER] = (
        TreesGeneratorBackendType.REMASTER
    )
    trajectory_attrs: dict[str, str] | None = None

    def _generate_one(
        self,
        rates: Rates,
        populations: list[str],
        init_values: list[int],
        output_file: str,
    ) -> None:
        temp_nexus_file = f"{output_file}-temp.nex"
        temp_xml_file = f"{output_file}-temp.xml"
        prepare_config_file(
            rates=rates,
            populations=populations,
            init_values=init_values,
            output_tree_file=temp_nexus_file,
            output_xml_file=temp_xml_file,
            trajectory_attrs=self.trajectory_attrs,
        )

        subprocess.run(
            ["beast", temp_xml_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        extract_newick_from_nexus(temp_nexus_file, output_file)
        process_newick_taxa_names(output_file, output_file, ["type", "time"])
        subprocess.run(["rm", temp_nexus_file], check=True)
        subprocess.run(["rm", temp_xml_file], check=True)


class TreesimulatorGenerator(BaseTreesGenerator):
    backend: Literal[TreesGeneratorBackendType.TREESIMULATOR] = (
        TreesGeneratorBackendType.TREESIMULATOR
    )
    kwargs: dict[str, Any] = Field(default_factory=dict)

    def _generate_one(
        self,
        rates: Rates,
        populations: list[str],
        init_values: list[int],
        output_file: str,
    ) -> None:
        generate_tree(
            rates=rates,
            populations=populations,
            output_file=output_file,
            **self.kwargs,
        )


TreesGenerator = Annotated[
    ReMASTERGenerator | TreesimulatorGenerator, Field(discriminator="backend")
]
