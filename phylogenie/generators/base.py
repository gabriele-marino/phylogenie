from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel


class GeneratorType(str, Enum):
    TREES = "trees"
    MSAS = "msas"


class BaseGenerator(ABC, BaseModel):
    type: GeneratorType

    @abstractmethod
    def generate(
        self,
        n_samples: int,
        output_dir: str,
        output_metadata_file: str,
        n_jobs: int,
    ) -> None: ...
