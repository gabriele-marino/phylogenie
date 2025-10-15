import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from numpy.random import default_rng
from tqdm import tqdm

from phylogenie.treesimulator.events import Event
from phylogenie.treesimulator.features import Feature, set_features
from phylogenie.treesimulator.io import dump_newick
from phylogenie.treesimulator.model import Model
from phylogenie.treesimulator.tree import Tree


def simulate_tree(
    events: Sequence[Event],
    n_tips: int | None = None,
    max_time: float = np.inf,
    init_state: str | None = None,
    sampling_probability_at_present: float = 0.0,
    seed: int | None = None,
    timeout: float = np.inf,
    acceptance_criterion: Callable[[Tree], bool] | None = None,
) -> tuple[Tree, dict[str, Any]]:
    if (max_time != np.inf) == (n_tips is not None):
        raise ValueError("Exactly one of max_time or n_tips must be specified.")
    if sampling_probability_at_present and max_time == np.inf:
        raise ValueError(
            "sampling_probability_at_present can only be used with max_time."
        )

    states = {e.state for e in events if e.state}
    if init_state is None and len(states) > 1:
        raise ValueError(
            "Init state must be provided for models with more than one state."
        )
    elif init_state is None:
        (init_state,) = states
    elif init_state not in states:
        raise ValueError(f"Init state {init_state} not found in event states: {states}")

    rng = default_rng(seed)
    start_clock = time.perf_counter()
    while True:
        model = Model(init_state)
        metadata: dict[str, Any] = {}
        run_events = list(events)
        current_time = 0.0
        change_times = sorted(set(t for e in events for t in e.rate.change_times))
        next_change_time = change_times.pop(0) if change_times else np.inf

        while current_time < max_time and (n_tips is None or model.n_sampled < n_tips):
            if time.perf_counter() - start_clock > timeout:
                raise TimeoutError("Simulation timed out.")

            rates = [e.get_propensity(model, current_time) for e in run_events]
            if not any(rates):
                break

            time_step = rng.exponential(1 / sum(rates))
            if current_time + time_step >= next_change_time:
                current_time = next_change_time
                next_change_time = change_times.pop(0) if change_times else np.inf
                continue
            if current_time + time_step >= max_time:
                current_time = max_time
                break
            current_time += time_step

            event_idx = np.searchsorted(np.cumsum(rates) / sum(rates), rng.random())
            event = run_events[int(event_idx)]
            event_metadata = event.apply(model, run_events, current_time, rng)
            if event_metadata is not None:
                metadata.update(event_metadata)

        if current_time != max_time and model.n_sampled != n_tips:
            continue

        for individual in model.get_population():
            if rng.random() < sampling_probability_at_present:
                model.sample(individual, current_time, True)

        tree = model.get_sampled_tree()
        if acceptance_criterion is None or acceptance_criterion(tree):
            return (tree, metadata)


def generate_trees(
    output_dir: str | Path,
    n_trees: int,
    events: Sequence[Event],
    n_tips: int | None = None,
    max_time: float = np.inf,
    init_state: str | None = None,
    sampling_probability_at_present: float = 0.0,
    node_features: Iterable[Feature] | None = None,
    seed: int | None = None,
    n_jobs: int = -1,
    timeout: float = np.inf,
    acceptance_criterion: Callable[[Tree], bool] | None = None,
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
                    n_tips=n_tips,
                    max_time=max_time,
                    init_state=init_state,
                    sampling_probability_at_present=sampling_probability_at_present,
                    seed=seed,
                    timeout=timeout,
                    acceptance_criterion=acceptance_criterion,
                )
                metadata["file_id"] = i
                if node_features is not None:
                    set_features(tree, node_features)
                dump_newick(tree, output_dir / f"{i}.nwk")
                return metadata
            except TimeoutError:
                print("Simulation timed out, retrying with a different seed...")
            seed += 1

    rng = default_rng(seed)
    jobs = joblib.Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        joblib.delayed(_simulate_tree)(i=i, seed=int(rng.integers(2**32)))
        for i in range(n_trees)
    )

    return pd.DataFrame(
        [md for md in tqdm(jobs, f"Generating trees in {output_dir}...", n_trees)]
    )
