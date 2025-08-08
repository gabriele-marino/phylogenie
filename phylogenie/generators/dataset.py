import os
from abc import ABC, abstractmethod
from enum import Enum
from itertools import product
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng
from tqdm import tqdm

import phylogenie.generators.configs as cfg
from phylogenie.generators.factories import eval_expression


class DataType(str, Enum):
    TREES = "trees"
    MSAS = "msas"


DATA_DIRNAME = "data"
METADATA_FILENAME = "metadata.csv"


class DatasetGenerator(ABC, cfg.StrictBaseModel):
    output_dir: str = "phylogenie-outputs"
    n_samples: int | dict[str, int] = 1
    n_jobs: int = -1
    seed: int | None = None
    context: dict[str, cfg.Distribution] | None = None

    @abstractmethod
    def _generate_one(
        self, filename: str, rng: Generator, data: dict[str, Any]
    ) -> None: ...

    def generate_one(
        self, filename: str, data: dict[str, Any] | None = None, seed: int | None = None
    ) -> None:
        data = {} if data is None else data
        self._generate_one(filename=filename, rng=default_rng(seed), data=data)

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

        data: list[dict[str, Any]] = [{} for _ in range(n_samples)]
        if self.context is not None:
            for d, (k, v) in product(data, self.context.items()):
                args = v.model_extra if v.model_extra is not None else {}
                for arg_name, arg_value in args.items():
                    if isinstance(arg_value, str):
                        args[arg_name] = eval_expression(arg_value, d)
                d[k] = np.array(getattr(rng, v.type)(**args)).tolist()
            df = pd.DataFrame([{"file_id": str(i), **d} for i, d in enumerate(data)])
            df.to_csv(os.path.join(output_dir, METADATA_FILENAME), index=False)

        jobs = joblib.Parallel(n_jobs=self.n_jobs, return_as="generator_unordered")(
            joblib.delayed(self.generate_one)(
                filename=os.path.join(data_dir, str(i)),
                data=data[i],
                seed=int(rng.integers(2**32)),
            )
            for i in range(n_samples)
        )
        for _ in tqdm(jobs, total=n_samples, desc=f"Generating {data_dir}..."):
            pass

    def generate(self) -> None:
        rng = default_rng(self.seed)
        if isinstance(self.n_samples, dict):
            for key, n_samples in self.n_samples.items():
                output_dir = os.path.join(self.output_dir, key)
                self._generate(rng, n_samples, output_dir)
        else:
            self._generate(rng, self.n_samples, self.output_dir)
