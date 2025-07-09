import os
from abc import ABC, abstractmethod
from enum import Enum

import joblib
import pandas as pd
from numpy.random import Generator, default_rng
from tqdm import tqdm

from phylogenie.configs import StrictBaseModel
from phylogenie.core.context import ContextConfig, context_factory
from phylogenie.core.typings import Data


class DataType(str, Enum):
    TREES = "trees"
    MSAS = "msas"


class DatasetGenerator(ABC, StrictBaseModel):
    output_dir: str = "phylogenie-out"
    n_samples: int | dict[str, int] = 1
    context: ContextConfig | None = None
    n_jobs: int = -1
    seed: int | None = None
    data_dir: str = "data"
    metadata_filename: str = "metadata.csv"

    @abstractmethod
    def _generate_one(self, filename: str, rng: Generator, data: Data) -> None: ...

    def generate_one(
        self, filename: str, data: Data | None = None, seed: int | None = None
    ) -> None:
        data = {} if data is None else data
        self._generate_one(filename=filename, rng=default_rng(seed), data=data)

    def _generate(self, rng: Generator, n_samples: int, output_dir: str) -> None:
        data_dir = os.path.join(output_dir, self.data_dir)
        metadata_file = os.path.join(output_dir, self.metadata_filename)
        if os.path.exists(data_dir):
            print(f"Output directory {data_dir} already exists. Skipping.")
            return
        os.makedirs(data_dir)

        data = [
            {} if self.context is None else context_factory(self.context, rng)
            for _ in range(n_samples)
        ]

        joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self.generate_one)(
                filename=os.path.join(data_dir, str(i)),
                data=d,
                seed=int(rng.integers(0, 2**32)),
            )
            for i, d in tqdm(
                enumerate(data), total=n_samples, desc=f"Generating {data_dir}..."
            )
        )

        df = pd.DataFrame([{"file_id": str(i), **d} for i, d in enumerate(data)])
        df.to_csv(metadata_file, index=False)

    def generate(self) -> None:
        rng = default_rng(self.seed)
        if isinstance(self.n_samples, dict):
            for key, n_samples in self.n_samples.items():
                output_dir = os.path.join(self.output_dir, key)
                self._generate(rng, n_samples, output_dir)
        else:
            self._generate(rng, self.n_samples, self.output_dir)
