from collections.abc import Sequence

from ete3 import TreeNode

STATE: str

def save_forest(forest: Sequence[TreeNode], nwk: str) -> None: ...
