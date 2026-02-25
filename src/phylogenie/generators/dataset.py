import os
from abc import ABC, abstractmethod
from random import Random
from typing import Annotated, Any

import joblib
import pandas as pd
from numpy.random import default_rng
from pydantic import Field
from tqdm import tqdm

import phylogenie.generators.configs as cfg
from phylogenie.generators.factories import context
from phylogenie.utils import Registry

DATA_DIRNAME = "data"
METADATA_FILENAME = "metadata.csv"


class DatasetGenerator(ABC, cfg.StrictBaseModel):
    output_dir: str = "phylogenie-outputs"
    n_samples: int | dict[str, int] = 1
    n_jobs: int = -1
    seed: int | None = None
    context: cfg.Context = Field(default_factory=dict)

    @abstractmethod
    def _generate_from_context(
        self, data_dir: str, file_id: str, context: dict[str, Any], seed: int
    ) -> dict[str, Any]: ...

    def _generate_one(self, data_dir: str, file_id: str, seed: int) -> dict[str, Any]:
        rng = default_rng(seed)
        while True:
            try:
                ctx = context(self.context, rng)
                metadata = self._generate_from_context(data_dir, file_id, ctx, seed)
                return {"file_id": file_id, **ctx, **metadata}
            except TimeoutError:
                print("Simulation timed out. Retrying with different context...")

    def _generate(self, n_samples: int, output_dir: str, rng: Random):
        if os.path.exists(output_dir):
            print(f"Output directory {output_dir} already exists. Skipping.")
            return

        data_dir = os.path.join(output_dir, DATA_DIRNAME)
        os.makedirs(data_dir)

        jobs = joblib.Parallel(n_jobs=self.n_jobs, return_as="generator_unordered")(
            joblib.delayed(self._generate_one)(data_dir, str(i), rng.getrandbits(32))
            for i in range(n_samples)
        )
        df = pd.DataFrame(
            [j for j in tqdm(jobs, f"Generating {data_dir}...", n_samples)]
        )
        df.to_csv(os.path.join(output_dir, METADATA_FILENAME), index=False)

    def generate(self):
        rng = Random(self.seed)
        if isinstance(self.n_samples, dict):
            for key, n_samples in self.n_samples.items():
                output_dir = os.path.join(self.output_dir, key)
                self._generate(n_samples, output_dir, rng)
        else:
            self._generate(self.n_samples, self.output_dir, rng)


DatasetGeneratorRegistry: Registry["DatasetGenerator"] = Registry(
    DatasetGenerator, "data_type"
)
DatasetGeneratorConfig = Annotated[
    "DatasetGenerator", DatasetGeneratorRegistry.Validator
]
