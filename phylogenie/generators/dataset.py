import os
from typing import Annotated

from pydantic import BaseModel, Field

from phylogenie.generators.msas import MSAsGenerator
from phylogenie.generators.trees import TreesGenerator
from phylogenie.utils.mixins import YAMLMixin

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
                output_dir=os.path.join(self.output_dir, DATA_DIR_NAME),
                output_metadata_file=os.path.join(self.output_dir, METADATA_FILE_NAME),
                n_samples=self.n_samples,
                n_jobs=self.n_jobs,
            )
        else:
            for split, n in self.n_samples.items():
                output_dir = os.path.join(self.output_dir, split)
                self.generator.generate(
                    output_dir=os.path.join(output_dir, DATA_DIR_NAME),
                    output_metadata_file=os.path.join(output_dir, METADATA_FILE_NAME),
                    n_samples=n,
                    n_jobs=self.n_jobs,
                )
