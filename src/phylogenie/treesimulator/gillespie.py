import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, TypeVar

import joblib
import numpy as np
import pandas as pd
from numpy.random import default_rng
from tqdm import tqdm

from phylogenie.io import dump_newick
from phylogenie.tree_node import TreeNode
from phylogenie.treesimulator.events import Event, TimedEvent
from phylogenie.treesimulator.models import Model

M = TypeVar("M", bound="Model")


def simulate_tree(
    events: Sequence[Event[M]],
    model: M,
    timed_events: Sequence[TimedEvent[M]] | None = None,
    n_leaves: int | None = None,
    max_time: float | None = None,
    seed: int | None = None,
    timeout: float | None = None,
    acceptance_criterion: Callable[[TreeNode], bool] | None = None,
    logs: dict[str, Callable[[TreeNode], Any]] | None = None,
) -> tuple[TreeNode, dict[str, Any]]:
    if (max_time is None) == (n_leaves is None):
        raise ValueError("Exactly one of max_time or n_leaves must be specified.")

    events_at_time: dict[float, list[TimedEvent[M]]] = defaultdict(list)
    if timed_events is not None:
        for te in timed_events:
            for t in te.times:
                events_at_time[t].append(te)

    rng = default_rng(seed)
    start_clock = time.perf_counter()
    while True:
        model.init()
        current_time = 0.0
        change_times = sorted(set(t for e in events for t in e.rate.change_times))
        next_change_time = change_times.pop(0) if change_times else None
        timed_event_times = sorted(events_at_time.keys())
        next_timed_event_time = timed_event_times.pop(0) if timed_event_times else None

        while n_leaves is None or model.tree_size < n_leaves:
            if timeout is not None and time.perf_counter() - start_clock > timeout:
                raise TimeoutError("Simulation timed out.")

            propensities = [e.get_propensity(model, current_time) for e in events]
            total_propensity = sum(propensities)
            next_event_time = (
                current_time + rng.exponential(1 / total_propensity)
                if total_propensity
                else None
            )

            if (
                next_change_time is not None
                and (next_event_time is None or next_change_time < next_event_time)
                and (
                    next_timed_event_time is None
                    or next_change_time <= next_timed_event_time
                )
                and (max_time is None or next_change_time < max_time)
            ):  # The next event is a rate change
                current_time = next_change_time
                next_change_time = change_times.pop(0) if change_times else None
            elif (
                next_timed_event_time is not None
                and (next_event_time is None or next_timed_event_time < next_event_time)
                and (max_time is None or next_timed_event_time <= max_time)
            ):  # The next event is a timed event
                current_time = next_timed_event_time
                for te in events_at_time[current_time]:
                    te.apply(model, current_time, rng)
                next_timed_event_time = (
                    timed_event_times.pop(0) if timed_event_times else None
                )
            elif next_event_time is not None and (
                max_time is None or next_event_time < max_time
            ):  # The next event is a stochastic event
                current_time = next_event_time
                event_idx = np.searchsorted(
                    np.cumsum(propensities) / total_propensity, rng.random()
                )
                event = events[int(event_idx)]
                event.apply(model, current_time, rng)
            else:  # No more events can occur
                break

        # If the simulation stopped because no more events could occur, restart
        if n_leaves is not None and model.tree_size < n_leaves:
            continue

        tree = model.get_tree()

        if (
            tree is None
            or acceptance_criterion is not None
            and not acceptance_criterion(tree)
        ):
            continue

        metadata: dict[str, Any] = {}
        if logs is not None:
            for key, func in logs.items():
                metadata[key] = func(tree)

        return (tree, metadata)


def generate_trees(
    output_dir: str | Path,
    n_trees: int,
    events: Sequence[Event[M]],
    model: M,
    timed_events: Sequence[TimedEvent[M]] | None = None,
    n_leaves: int | None = None,
    max_time: float | None = None,
    node_features: Mapping[str, str] | None = None,
    seed: int | None = None,
    n_jobs: int = -1,
    timeout: float | None = None,
    acceptance_criterion: Callable[[TreeNode], bool] | None = None,
    logs: dict[str, Callable[[TreeNode], Any]] | None = None,
) -> pd.DataFrame:
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory {output_dir} already exists")
    output_dir.mkdir(parents=True)

    def _simulate_tree(i: int, seed: int) -> dict[str, Any]:
        while True:
            try:
                tree, metadata = simulate_tree(
                    events=events,
                    n_leaves=n_leaves,
                    max_time=max_time,
                    model=model,
                    timed_events=timed_events,
                    seed=seed,
                    timeout=timeout,
                    acceptance_criterion=acceptance_criterion,
                    logs=logs,
                )
                metadata["file_id"] = i
                if node_features is not None:
                    for name, feature in node_features.items():
                        mapping = getattr(tree, feature)
                        for node in tree:
                            node[name] = mapping[node]
                dump_newick(tree, output_dir / f"{i}.nwk")
                return metadata
            except TimeoutError as e:
                print(f"{e}. Retrying with a different seed...")
            seed += 1

    rng = default_rng(seed)
    jobs = joblib.Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        joblib.delayed(_simulate_tree)(i=i, seed=int(rng.integers(2**32)))
        for i in range(n_trees)
    )

    return pd.DataFrame(
        [md for md in tqdm(jobs, f"Generating trees in {output_dir}...", n_trees)]
    )
