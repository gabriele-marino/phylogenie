import numpy as np
from ete3.coretype.tree import TreeNode
from treesimulator.mtbd_models import Model

def generate(
    model: Model | list[Model],
    min_tips: int,
    max_tips: int,
    T: float = np.inf,
    skyline_times: list[float] | None = None,
    state_frequencies: list[float] | None = None,
    max_notified_contacts: int = 1,
) -> tuple[list[TreeNode], tuple[int, int, float], dict[float, int]]: ...
