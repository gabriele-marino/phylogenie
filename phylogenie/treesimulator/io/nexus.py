import re
from collections.abc import Iterator
from pathlib import Path

from phylogenie.treesimulator.io.newick import parse_newick
from phylogenie.treesimulator.tree import Tree


def _parse_translate_block(lines: Iterator[str]) -> dict[str, str]:
    translations: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        match = re.match(r"(\d+)\s+['\"]?([^'\",;]+)['\"]?", line)
        if match is None:
            if ";" in line:
                return translations
            else:
                raise ValueError(f"Invalid translate line. Expected '<num> <name>'.")
        translations[match.group(1)] = match.group(2)
    raise ValueError("Translate block not terminated with ';'.")


def _parse_trees_block(lines: Iterator[str]) -> dict[str, Tree]:
    trees: dict[str, Tree] = {}
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
                    f"Invalid tree line. Expected 'TREE <name> = <newick>'."
                )
            name = match.group(1)
            if name in trees:
                raise ValueError(f"Duplicate tree name found: {name}.")
            trees[name] = parse_newick(match.group(2), translations)
    raise ValueError("Unterminated TREES block.")


def load_nexus(nexus_file: str | Path) -> dict[str, Tree]:
    with open(nexus_file, "r") as f:
        for line in f:
            if line.strip().upper() == "BEGIN TREES;":
                return _parse_trees_block(f)
    raise ValueError("No TREES block found in the NEXUS file.")
