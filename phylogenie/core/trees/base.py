from enum import Enum
from typing import Literal

from phylogenie.core.dataset import DatasetGenerator, DataType


class BackendType(str, Enum):
    REMASTER = "remaster"
    TREESIMULATOR = "treesimulator"


class TreesGenerator(DatasetGenerator):
    data_type: Literal[DataType.TREES] = DataType.TREES
