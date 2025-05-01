from typing import Callable, Iterator

class TreeNode(object):
    name: str
    def iter_leaves(
        self, is_leaf_fn: Callable[[TreeNode], bool] | None = None
    ) -> Iterator[TreeNode]: ...
    def get_distance(
        self,
        target: TreeNode,
        target2: TreeNode | None = None,
        topology_only: bool = False,
    ) -> float: ...
    def get_leaves(self) -> list[TreeNode]: ...
