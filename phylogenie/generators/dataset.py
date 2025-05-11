import os
from typing import Annotated

from pydantic import BaseModel, Field
from pykit.mixins import YAMLMixin

from phylogenie.generators.msas import MSAsGenerator
from phylogenie.generators.trees import TreesGenerator

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
