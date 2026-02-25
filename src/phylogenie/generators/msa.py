import os
from abc import ABC, abstractmethod
from typing import Annotated, Any

import phylogenie.generators.configs as cfg
from phylogenie.generators.dataset import DatasetGenerator, DatasetGeneratorRegistry
from phylogenie.generators.tree import TreeGeneratorConfig
from phylogenie.io import dump_newick, load_newick
from phylogenie.tree_node import TreeNode
from phylogenie.utils import Registry

MSAS_DIRNAME = "MSAs"
TREES_DIRNAME = "trees"


class MSAGenerator(ABC, cfg.StrictBaseModel):
    @abstractmethod
    def generate(
        self,
        filename: str,
        input_tree_file: str,
        context: dict[str, Any],
        seed: int | None = None,
    ) -> None: ...


MSAGeneratorRegistry: Registry["MSAGenerator"] = Registry(MSAGenerator)
MSAGeneratorConfig = Annotated[MSAGenerator, MSAGeneratorRegistry.Validator]


@DatasetGeneratorRegistry.register("msa")
class MSADatasetGenerator(DatasetGenerator):
    tree_generator: TreeGeneratorConfig
    msa_generator: MSAGeneratorConfig
    keep_trees: bool = False

    def _generate_from_context(
        self, data_dir: str, file_id: str, context: dict[str, Any], seed: int
    ) -> dict[str, Any]:
        trees_dir = (
            os.path.join(data_dir, TREES_DIRNAME) if self.keep_trees else data_dir
        )
        msas_dir = os.path.join(data_dir, MSAS_DIRNAME) if self.keep_trees else data_dir

        os.makedirs(trees_dir, exist_ok=True)
        os.makedirs(msas_dir, exist_ok=True)

        tree_filename = os.path.join(trees_dir, file_id)
        msa_filename = os.path.join(msas_dir, file_id)

        metadata = self.tree_generator.generate(tree_filename, context, seed)

        tree_file = f"{tree_filename}.nwk"
        tree = load_newick(tree_file)
        assert isinstance(tree, TreeNode)

        times = tree.times
        for leaf in tree.get_leaves():
            leaf.name += f"|{times[leaf]}"
        dump_newick(tree, tree_file)

        self.msa_generator.generate(msa_filename, tree_file, context, seed)
        if not self.keep_trees:
            os.remove(tree_file)

        return metadata
