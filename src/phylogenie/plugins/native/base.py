from abc import abstractmethod
from typing import Annotated, Any, Callable

from numpy.random import Generator, default_rng
from pydantic import Field

import phylogenie.generators.configs as cfg
import phylogenie.generators.factories as f
from phylogenie.generators.tree import TreeGenerator
from phylogenie.io import dump_newick
from phylogenie.plugins.native.timed_events import TimedEventConfig
from phylogenie.tree_node import TreeNode
from phylogenie.treesimulator import Model, simulate_tree


class PhylogenieTreeGenerator(TreeGenerator):
    n_leaves: cfg.Integer | None = None
    max_time: cfg.Scalar | None = None
    timeout: float | None = None
    acceptance_criterion: str | None = None
    tree_logs: Annotated[dict[str, str], Field(default_factory=dict)]
    model_logs: Annotated[dict[str, str], Field(default_factory=dict)]
    timed_events: Annotated[list[TimedEventConfig], Field(default_factory=list)]

    @abstractmethod
    def _get_model(self, context: dict[str, Any], rng: Generator) -> Model: ...

    def generate(
        self, filename: str, context: dict[str, Any], seed: int | None = None
    ) -> dict[str, Any]:
        rng = default_rng(seed)
        model = self._get_model(context, rng)
        model.rng.seed(seed)

        for event in self.timed_events:
            model.add_event(event.factory(context, rng))

        max_time = None if self.max_time is None else f.scalar(self.max_time, context)
        n_leaves = None if self.n_leaves is None else f.integer(self.n_leaves, context)

        acceptance_criterion: None | Callable[[TreeNode], bool] = (
            None
            if self.acceptance_criterion is None
            else lambda tree: f.eval_expression(
                self.acceptance_criterion,  # pyright: ignore
                context,
                extra_context={"tree": tree},
            )
        )

        def _tree_logs(tree: TreeNode) -> dict[str, Any]:
            return {
                key: f.eval_expression(expr, context, extra_context={"tree": tree})
                for key, expr in self.tree_logs.items()
            }

        def _model_logs(model: Model) -> dict[str, Any]:
            return {
                key: f.eval_expression(expr, context, extra_context={"model": model})
                for key, expr in self.model_logs.items()
            }

        tree, metadata = simulate_tree(
            model=model,
            n_leaves=n_leaves,
            max_time=max_time,
            timeout=self.timeout,
            acceptance_criterion=acceptance_criterion,
            tree_logs=_tree_logs,
            model_logs=_model_logs,
        )

        dump_newick(tree, f"{filename}.nwk")
        return metadata
