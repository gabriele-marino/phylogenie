import os
import time
from collections.abc import Iterable, Sequence
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.random import default_rng
from tqdm import tqdm

from phylogenie.io import dump_newick
from phylogenie.tree import Tree
from phylogenie.treesimulator.events import Event
from phylogenie.treesimulator.features import Feature, set_features
from phylogenie.treesimulator.model import Model


def simulate_tree(
    events: Sequence[Event],
    min_tips: int = 1,
    max_tips: int | None = None,
    max_time: float = np.inf,
    init_state: str | None = None,
    sampling_probability_at_present: float = 0.0,
    seed: int | None = None,
    timeout: float = np.inf,
) -> tuple[Tree, dict[str, Any]]:
    if max_time == np.inf and sampling_probability_at_present:
        raise ValueError(
            "sampling_probability_at_present cannot be set when max_time is infinite."
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

        if max_time == np.inf:
            if max_tips is None:
                raise ValueError("Either max_time or max_tips must be specified.")
            target_n_tips = rng.integers(min_tips, max_tips + 1)
        else:
            target_n_tips = None

        while current_time < max_time:
            if time.perf_counter() - start_clock > timeout:
                raise TimeoutError("Simulation timed out.")

            rates = [e.get_propensity(model, current_time) for e in run_events]

            instantaneous_events = [e for e, r in zip(run_events, rates) if r == np.inf]
            if instantaneous_events:
                event = instantaneous_events[rng.integers(len(instantaneous_events))]
                event.apply(model, run_events, current_time, rng)
                continue

            if (
                not any(rates)
                or max_tips is not None
                and model.n_sampled >= max_tips
                or target_n_tips is not None
                and model.n_sampled >= target_n_tips
            ):
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

        for individual in model.get_population():
            if rng.random() < sampling_probability_at_present:
                model.sample(individual, current_time, True)

        if min_tips <= model.n_sampled and (
            max_tips is None or model.n_sampled <= max_tips
        ):
            return (model.get_sampled_tree(), metadata)


def generate_trees(
    output_dir: str,
    n_trees: int,
    events: Sequence[Event],
    min_tips: int = 1,
    max_tips: int | None = None,
    max_time: float = np.inf,
    init_state: str | None = None,
    sampling_probability_at_present: float = 0.0,
    node_features: Iterable[Feature] | None = None,
    seed: int | None = None,
    n_jobs: int = -1,
    timeout: float = np.inf,
) -> pd.DataFrame:
    def _simulate_tree(seed: int) -> tuple[Tree, dict[str, Any]]:
        while True:
            try:
                tree, metadata = simulate_tree(
                    events=events,
                    min_tips=min_tips,
                    max_tips=max_tips,
                    max_time=max_time,
                    init_state=init_state,
                    sampling_probability_at_present=sampling_probability_at_present,
                    seed=seed,
                    timeout=timeout,
                )
                if node_features is not None:
                    set_features(tree, node_features)
                return (tree, metadata)
            except TimeoutError:
                print("Simulation timed out, retrying with a different seed...")
            seed += 1

    if os.path.exists(output_dir):
        raise FileExistsError(f"Output directory {output_dir} already exists")
    os.makedirs(output_dir)

    rng = default_rng(seed)
    jobs = joblib.Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        joblib.delayed(_simulate_tree)(seed=int(rng.integers(2**32)))
        for _ in range(n_trees)
    )

    df: list[dict[str, Any]] = []
    for i, (tree, metadata) in tqdm(
        enumerate(jobs), total=n_trees, desc=f"Generating trees in {output_dir}..."
    ):
        df.append({"file_id": i} | metadata)
        dump_newick(tree, os.path.join(output_dir, f"{i}.nwk"))
    return pd.DataFrame(df)
