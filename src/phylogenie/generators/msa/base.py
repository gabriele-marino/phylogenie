from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import phylogenie.generators._configs as cfg


class Backend(str, Enum):
    ALISIM = "alisim"


class MSAGenerator(ABC, cfg.StrictBaseModel):
    @abstractmethod
    def generate(
        self,
        filename: str,
        input_tree_file: str,
        context: dict[str, Any],
        seed: int | None = None,
    ) -> None: ...
