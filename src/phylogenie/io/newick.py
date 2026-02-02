import re
from pathlib import Path

from phylogenie.tree_node import TreeNode


def parse_newick(newick: str, translations: dict[str, str] | None = None) -> TreeNode:
    """
    Parse a Newick string into a TreeNode object.

    The parser supports embedded metadata in the `[&key=value]` syntax and an
    optional translation mapping for taxa names.

    Parameters
    -----------
    newick : str
        The Newick-formatted string to parse.
    translations : dict[str, str] | None, optional
        Optional translation mapping (e.g., from NEXUS translate blocks).

    Returns
    --------
    TreeNode
        The parsed tree.
    """
    newick = newick.strip()
    newick = re.sub(r"^\[\&[^\]]*\]", "", newick).strip()

    stack: list[list[TreeNode]] = []
    current_children: list[TreeNode] = []
    current_nodes: list[TreeNode] = []
    i = 0
    while True:

        def _read_chars(stoppers: list[str]) -> str:
            nonlocal i
            chars = ""
            while i < len(newick) and newick[i] not in stoppers:
                chars += newick[i]
                i += 1
            if i == len(newick):
                raise ValueError(f"Expected one of {stoppers}, got end of string")
            return chars

        if newick[i] == "(":
            stack.append(current_nodes)
            current_nodes = []
            i += 1
            continue

        name = _read_chars([":", "[", ",", ")", ";"])
        if translations is not None and name in translations:
            name = translations[name]
        current_node = TreeNode(name)

        if newick[i] == "[":
            i += 1
            if newick[i] != "&":
                raise ValueError("Expected '[&' at the start of node features")
            i += 1
            features = re.split(r",(?=[^,]+=)", _read_chars(["]"]))
            i += 1
            for feature in features:
                key, value = feature.split("=")
                try:
                    current_node.set(key, eval(value))
                except Exception:
                    current_node.set(key, value)

        if newick[i] == ":":
            i += 1
            current_node.branch_length = float(_read_chars([",", ")", ";"]))

        for node in current_children:
            current_node.add_child(node)
            current_children = []
        current_nodes.append(current_node)

        if newick[i] == ")":
            current_children = current_nodes
            current_nodes = stack.pop()
        elif newick[i] == ";":
            return current_node

        i += 1


def load_newick(filepath: str | Path) -> TreeNode | list[TreeNode]:
    """
    Load one or more Newick trees from a file.

    If the file contains a single tree, a TreeNode is returned. If it contains
    multiple trees (one per line), a list of TreeNode objects is returned.

    Parameters
    -----------
    filepath : str | Path
        Path to the Newick file to read.

    Returns
    --------
    TreeNode | list[TreeNode]
        A single tree or a list of trees depending on the file contents.
    """
    with open(filepath, "r") as file:
        trees = [parse_newick(newick) for newick in file]
    return trees[0] if len(trees) == 1 else trees


def to_newick(node: TreeNode) -> str:
    """
    Serialize a TreeNode object to a Newick string.

    Metadata keys are encoded using the `[&key=value]` syntax. Certain
    characters in keys or values are rejected to avoid invalid Newick.

    Parameters
    -----------
    node : TreeNode
        The tree node to serialize.

    Returns
    --------
    str
        The Newick representation of the tree (without trailing semicolon).
    """

    children_newick = ",".join([to_newick(child) for child in node.children])
    newick = node.name
    if node.metadata:
        reprs = {k: repr(v).replace("'", '"') for k, v in node.metadata.items()}
        for k, r in reprs.items():
            if "," in k or "=" in k or "]" in k:
                raise ValueError(
                    f"Invalid feature key `{k}`: keys must not contain ',', '=', or ']'"
                )
            if "=" in r or "]" in r:
                raise ValueError(
                    f"Invalid value  `{r}` for feature `{k}`: values must not contain '=' or ']'"
                )
        features = [f"{k}={r}" for k, r in reprs.items()]
        newick += f"[&{','.join(features)}]"
    if children_newick:
        newick = f"({children_newick}){newick}"
    if node.branch_length is not None:
        newick += f":{node.branch_length}"
    return newick


def dump_newick(trees: TreeNode | list[TreeNode], filepath: str | Path):
    """
    Write one or more trees to a Newick file.

    Each tree is written on its own line and terminated with a semicolon.

    Parameters
    -----------
    trees : TreeNode | list[TreeNode]
        The tree or trees to write.
    filepath : str | Path
        Output file path.
    """
    if isinstance(trees, TreeNode):
        trees = [trees]
    with open(filepath, "w") as file:
        for t in trees:
            file.write(to_newick(t) + ";\n")
