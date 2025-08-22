import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import joblib
import pandas as pd
from numpy.random import Generator, default_rng
from tqdm import tqdm

from phylogenie.utils import Distribution, StrictBaseModel


class DataType(str, Enum):
    TREES = "trees"
    MSAS = "msas"


DATA_DIRNAME = "data"
METADATA_FILENAME = "metadata.csv"


class DatasetGenerator(ABC, StrictBaseModel):
    output_dir: str = "phylogenie-outputs"
    n_samples: int | dict[str, int] = 1
    n_jobs: int = -1
    seed: int | None = None
    context: dict[str, Distribution] | None = None

    @abstractmethod
    def generate_one(
        self,
        filename: str,
        context: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]: ...

    def _generate(self, rng: Generator, n_samples: int, output_dir: str) -> None:
        if os.path.exists(output_dir):
            print(f"Output directory {output_dir} already exists. Skipping.")
            return

        data_dir = (
            output_dir
            if self.context is None
            else os.path.join(output_dir, DATA_DIRNAME)
        )
        os.makedirs(data_dir)

        jobs = joblib.Parallel(n_jobs=self.n_jobs, return_as="generator_unordered")(
            joblib.delayed(self.generate_one)(
                seed=int(rng.integers(2**32)),
                filename=os.path.join(data_dir, str(i)),
                context=self.context,
            )
            for i in range(n_samples)
        )
        df = pd.DataFrame(
            [r for r in tqdm(jobs, total=n_samples, desc=f"Generating {data_dir}...")]
        )
        df.to_csv(os.path.join(output_dir, METADATA_FILENAME), index=False)

    def generate(self) -> None:
        rng = default_rng(self.seed)
        if isinstance(self.n_samples, dict):
            for key, n_samples in self.n_samples.items():
                output_dir = os.path.join(self.output_dir, key)
                self._generate(rng, n_samples, output_dir)
        else:
            self._generate(rng, self.n_samples, self.output_dir)
