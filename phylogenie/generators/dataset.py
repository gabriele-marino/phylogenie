import os
from typing import Annotated

from kitpy.mixins import YAMLMixin
from pydantic import BaseModel, Field

from phylogenie.generators.msas import MSAsGenerator
from phylogenie.generators.trees import TreesGenerator

DATA_DIR_NAME = "data"
METADATA_FILE_NAME = "metadata.csv"

Generator = Annotated[
    MSAsGenerator | TreesGenerator,
    Field(
        discriminator="type",
    ),
]


class DatasetGenerator(BaseModel, YAMLMixin):
    output_dir: str
    n_samples: int | dict[str, int]
    generator: Generator
    n_jobs: int = -1

    def run(self) -> None:
        if isinstance(self.n_samples, int):
            self.generator.generate(
                output_dir=self.output_dir,
                n_samples=self.n_samples,
                n_jobs=self.n_jobs,
            )
        else:
            for split, n in self.n_samples.items():
                self.generator.generate(
                    output_dir=os.path.join(self.output_dir, split),
                    n_samples=n,
                    n_jobs=self.n_jobs,
                )
