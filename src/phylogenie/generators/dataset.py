import os
from abc import ABC, abstractmethod
from enum import Enum
from random import Random
from typing import Annotated, Any, Literal

import joblib
import pandas as pd
from pydantic import Field, PrivateAttr
from tqdm import tqdm

import phylogenie.generators._configs as cfg
from phylogenie.generators._factories import context
from phylogenie.generators.msa import MSAGeneratorConfig
from phylogenie.generators.tree import TreeGeneratorConfig
from phylogenie.io import dump_newick, load_newick
from phylogenie.tree_node import TreeNode


class DataType(str, Enum):
    TREE = "tree"
    MSA = "msa"


DATA_DIRNAME = "data"
MSAS_DIRNAME = "MSAs"
TREES_DIRNAME = "trees"
METADATA_FILENAME = "metadata.csv"


class DatasetGenerator(ABC, cfg.StrictBaseModel):
    output_dir: str = "phylogenie-outputs"
    n_samples: int | dict[str, int] = 1
    n_jobs: int = -1
    seed: int | None = None
    context: cfg.Context = Field(default_factory=dict)

    _rng: Random = PrivateAttr()

    def model_post_init(self, _: Any):
        self._rng = Random(self.seed)

    def _seed(self, seed: int | None):
        self._rng.seed(seed)

    def _draw_seed(self) -> int:
        return int(self._rng.randint(0, 2**32))

    @abstractmethod
    def _generate_from_context(
        self, data_dir: str, file_id: str, context: dict[str, Any]
    ) -> dict[str, Any]: ...

    def _generate_one(
        self, data_dir: str, file_id: str, seed: int | None = None
    ) -> dict[str, Any]:
        self._seed(seed)
        while True:
            try:
                ctx = context(self.context, seed=self._draw_seed())
                metadata = self._generate_from_context(data_dir, file_id, ctx)
                return {"file_id": file_id, **ctx, **metadata}
            except TimeoutError:
                print("Simulation timed out. Retrying with different context...")

    def _generate(self, n_samples: int, output_dir: str):
        if os.path.exists(output_dir):
            print(f"Output directory {output_dir} already exists. Skipping.")
            return

        data_dir = os.path.join(output_dir, DATA_DIRNAME)
        os.makedirs(data_dir)

        jobs = joblib.Parallel(n_jobs=self.n_jobs, return_as="generator_unordered")(
            joblib.delayed(self._generate_one)(
                data_dir=data_dir, file_id=str(i), seed=self._draw_seed()
            )
            for i in range(n_samples)
        )
        df = pd.DataFrame(
            [j for j in tqdm(jobs, f"Generating {data_dir}...", n_samples)]
        )
        df.to_csv(os.path.join(output_dir, METADATA_FILENAME), index=False)

    def generate(self):
        if isinstance(self.n_samples, dict):
            for key, n_samples in self.n_samples.items():
                output_dir = os.path.join(self.output_dir, key)
                self._generate(n_samples, output_dir)
        else:
            self._generate(self.n_samples, self.output_dir)


class TreeDatasetGenerator(DatasetGenerator):
    data_type: Literal[DataType.TREE] = DataType.TREE
    tree_generator: TreeGeneratorConfig

    def _generate_from_context(
        self, data_dir: str, file_id: str, context: dict[str, Any]
    ) -> dict[str, Any]:

        return self.tree_generator.generate(
            filename=os.path.join(data_dir, file_id),
            context=context,
            seed=self._draw_seed(),
        )


class MSADatasetGenerator(DatasetGenerator):
    data_type: Literal[DataType.MSA] = DataType.MSA
    tree_generator: TreeGeneratorConfig
    msa_generator: MSAGeneratorConfig
    keep_trees: bool = False

    def _generate_from_context(
        self, data_dir: str, file_id: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        trees_dir = (
            os.path.join(data_dir, TREES_DIRNAME) if self.keep_trees else data_dir
        )
        msas_dir = os.path.join(data_dir, MSAS_DIRNAME) if self.keep_trees else data_dir

        os.makedirs(trees_dir, exist_ok=True)
        os.makedirs(msas_dir, exist_ok=True)

        tree_filename = os.path.join(trees_dir, file_id)
        msa_filename = os.path.join(msas_dir, file_id)

        metadata = self.tree_generator.generate(
            filename=tree_filename, context=context, seed=self._draw_seed()
        )

        tree_file = f"{tree_filename}.nwk"
        tree = load_newick(tree_file)
        assert isinstance(tree, TreeNode)

        times = tree.times
        for leaf in tree.get_leaves():
            leaf.name += f"|{times[leaf]}"
        dump_newick(tree, tree_file)

        self.msa_generator.generate(
            filename=msa_filename,
            input_tree_file=tree_file,
            context=context,
            seed=self._draw_seed(),
        )
        if not self.keep_trees:
            os.remove(tree_file)

        return metadata


DatasetGeneratorConfig = Annotated[
    TreeDatasetGenerator | MSADatasetGenerator, Field(discriminator="data_type")
]
