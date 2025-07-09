from collections.abc import Sequence

import numpy as np
from ete3 import TreeNode
from treesimulator.mtbd_models import Model

def generate(
    model: Model | Sequence[Model],
    min_tips: int,
    max_tips: int,
    T: float = np.inf,
    skyline_times: Sequence[float] | None = None,
    state_frequencies: Sequence[float] | None = None,
    max_notified_contacts: int = 1,
    root_state: str | None = None,
    random_seed: int | None = None,
) -> tuple[list[TreeNode], tuple[int, int, float], dict[float, int]]: ...
