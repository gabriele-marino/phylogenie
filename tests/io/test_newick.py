from pathlib import Path

import pytest

from phylogenie.io.newick import dump_newick, load_newick, parse_newick, to_newick
from phylogenie.tree_node import TreeNode


def test_parse_newick_with_translation():
    tree = parse_newick("(1:0.1,2:0.2)root;", {"1": "A", "2": "B"})
    assert tree.name == "root"
    assert [child.name for child in tree.children] == ["A", "B"]
    assert [child.branch_length for child in tree.children] == [0.1, 0.2]


def test_parse_newick_with_metadata():
    tree = parse_newick('A[&rate=2,tag="x"];')
    assert tree.name == "A"
    assert tree["rate"] == 2
    assert tree["tag"] == "x"


def test_to_newick_roundtrip():
    root = TreeNode("root", branch_length=0.3)
    child_a = TreeNode("A", branch_length=0.1)
    child_a.set("rate", 2)
    child_b = TreeNode("B", branch_length=0.2)
    root.add_child(child_a)
    root.add_child(child_b)

    newick = to_newick(root)
    assert newick == "(A[&rate=2]:0.1,B:0.2)root:0.3;"

    parsed = parse_newick(newick)

    assert parsed.name == "root"
    assert [child.name for child in parsed.children] == ["A", "B"]
    assert parsed.get_descendant("A")["rate"] == 2


def test_to_newick_invalid_metadata_key():
    node = TreeNode("root")
    node.set("bad=key", 1)
    with pytest.raises(ValueError):
        to_newick(node)


def test_to_newick_invalid_metadata_value():
    node = TreeNode("root")
    node.set("tag", "a=b")
    with pytest.raises(ValueError):
        to_newick(node)


def test_dump_and_load_newick(tmp_path: Path):
    trees = [TreeNode("A"), TreeNode("B")]
    path = tmp_path / "trees.newick"

    dump_newick(trees, path)
    loaded = load_newick(path)

    assert all(loaded.name == tree.name for tree, loaded in zip(trees, loaded))
