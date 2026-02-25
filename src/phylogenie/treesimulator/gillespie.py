import time
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
from numpy.random import default_rng
from tqdm import tqdm

from phylogenie.io import dump_newick
from phylogenie.tree_node import TreeNode
from phylogenie.treesimulator.core import Model


def simulate_tree(
    model: Model,
    n_leaves: int | None = None,
    max_time: float | None = None,
    timeout: float | None = None,
    acceptance_criterion: Callable[[TreeNode], bool] | None = None,
    tree_logs: Callable[[TreeNode], dict[str, Any]] | None = None,
    model_logs: Callable[[Model], dict[str, Any]] | None = None,
) -> tuple[TreeNode, dict[str, Any]]:
    start_clock = time.perf_counter()
    while True:
        model.reset()
        while model.step(max_time) and (n_leaves is None or model.n_sampled < n_leaves):
            if timeout is not None and time.perf_counter() - start_clock > timeout:
                raise TimeoutError("Simulation timed out.")

        # If the simulation stopped because the process stopped,
        # and not because we reached the desired number of leaves,
        # we restart the simulation.
        if n_leaves is not None and model.n_sampled < n_leaves:
            continue

        tree = model.get_sampled_tree()

        if (
            tree is None
            or acceptance_criterion is not None
            and not acceptance_criterion(tree)
        ):
            continue

        metadata: dict[str, Any] = {}
        if tree_logs is not None:
            metadata.update(tree_logs(tree))
        if model_logs is not None:
            metadata.update(model_logs(model))

        return (tree, metadata)


def generate_trees(
    output_dir: str | Path,
    n_trees: int,
    model: Model,
    n_leaves: int | None = None,
    max_time: float | None = None,
    seed: int | None = None,
    n_jobs: int = -1,
    timeout: float | None = None,
    acceptance_criterion: Callable[[TreeNode], bool] | None = None,
    tree_logs: Callable[[TreeNode], dict[str, Any]] | None = None,
    model_logs: Callable[[Model], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory {output_dir} already exists")
    output_dir.mkdir(parents=True)

    def _simulate_tree(i: int, seed: int) -> dict[str, Any]:
        while True:
            try:
                model.rng.seed(seed)
                tree, metadata = simulate_tree(
                    model=model,
                    n_leaves=n_leaves,
                    max_time=max_time,
                    timeout=timeout,
                    acceptance_criterion=acceptance_criterion,
                    tree_logs=tree_logs,
                    model_logs=model_logs,
                )
                metadata["file_id"] = i
                dump_newick(tree, output_dir / f"{i}.nwk")
                return metadata
            except TimeoutError:
                print("Simulation timed out. Retrying with a different seed...")
            seed += 1

    rng = default_rng(seed)
    jobs = joblib.Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        joblib.delayed(_simulate_tree)(i=i, seed=int(rng.integers(2**32)))
        for i in range(n_trees)
    )

    return pd.DataFrame(
        [md for md in tqdm(jobs, f"Generating trees in {output_dir}...", n_trees)]
    )
