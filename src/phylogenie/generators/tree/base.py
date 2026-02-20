from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import phylogenie.generators._configs as cfg


class Backend(str, Enum):
    PHYLOGENIE = "phylogenie"


class TreeGenerator(ABC, cfg.StrictBaseModel):
    @abstractmethod
    def generate(
        self, filename: str, context: dict[str, Any], seed: int | None = None
    ) -> dict[str, Any]: ...
