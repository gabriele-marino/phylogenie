from pathlib import Path

import pytest

from phylogenie.io.nexus import load_nexus


def test_load_nexus_with_translate(tmp_path: Path):
    nexus = """
    #NEXUS
    BEGIN TREES;
    TRANSLATE
    1 A,
    
    2 B;
    TREE tree1 = (1:0.1,2:0.2)root;

    TREE tree2 = (1:0.3,2:0.4)root;
    END;
    """
    path = tmp_path / "trees.nex"
    path.write_text(nexus, encoding="utf-8")

    trees = load_nexus(path)

    assert list(trees.keys()) == ["tree1", "tree2"]
    tree = trees["tree1"]
    assert tree.name == "root"
    assert [child.name for child in tree.children] == ["A", "B"]


def test_load_nexus_without_trees_block(tmp_path: Path):
    path = tmp_path / "bad.nex"
    path.write_text(
        "#NEXUS\nBEGIN TAXA;\nDIMENSIONS NTAX=123;\nTAXLABELS\nTAXA1;\nEND;\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_nexus(path)
