import re
from collections.abc import Iterator
from pathlib import Path

from phylogenie.io.newick import parse_newick
from phylogenie.tree_node import TreeNode


def _parse_translate_block(lines: Iterator[str]) -> dict[str, str]:
    """
    Parse a TRANSLATE block from a NEXUS file.

    Parameters
    -----------
    lines : Iterator[str]
        Iterator over lines in the NEXUS file positioned after "TRANSLATE".

    Returns
    --------
    dict[str, str]
        Mapping from numeric IDs to taxa names.
    """
    translations: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        match = re.match(r"(\d+)\s+['\"]?([^'\",;]+)['\"]?", line)
        if match is None:
            if ";" in line:
                return translations
            else:
                raise ValueError("Invalid translate line. Expected '<num> <name>'.")
        translations[match.group(1)] = match.group(2)
    raise ValueError("Translate block not terminated with ';'.")


def _parse_trees_block(lines: Iterator[str]) -> dict[str, TreeNode]:
    """
    Parse a TREES block from a NEXUS file.

    Parameters
    -----------
    lines : Iterator[str]
        Iterator over lines in the NEXUS file positioned after "BEGIN TREES;".

    Returns
    --------
    dict[str, TreeNode]
        Mapping from tree names to parsed TreeNode objects.
    """
    trees: dict[str, TreeNode] = {}
    translations = {}
    for line in lines:
        line = line.strip()
        if line.upper() == "TRANSLATE":
            translations = _parse_translate_block(lines)
        elif line.upper() == "END;":
            return trees
        else:
            match = re.match(r"^TREE\s*\*?\s+(\S+)\s*=\s*(.+)$", line, re.IGNORECASE)
            if match is None:
                raise ValueError(
                    "Invalid tree line. Expected 'TREE <name> = <newick>'."
                )
            name = match.group(1)
            if name in trees:
                raise ValueError(f"Duplicate tree name found: {name}.")
            trees[name] = parse_newick(match.group(2), translations)
    raise ValueError("Unterminated TREES block.")


def load_nexus(nexus_file: str | Path) -> dict[str, TreeNode]:
    """
    Load trees from a NEXUS file.

    Only the TREES block is parsed. Tree names are preserved in the returned
    dictionary.

    Parameters
    -----------
    nexus_file : str | Path
        Path to the NEXUS file.

    Returns
    --------
    dict[str, TreeNode]
        Mapping of tree names to TreeNode objects.
    """
    with open(nexus_file, "r") as f:
        for line in f:
            if line.strip().upper() == "BEGIN TREES;":
                return _parse_trees_block(f)
    raise ValueError("No TREES block found in the NEXUS file.")
