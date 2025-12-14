import re
from pathlib import Path

from phylogenie.treesimulator.tree import Tree


def parse_newick(newick: str, translations: dict[str, str] | None = None) -> Tree:
    newick = newick.strip()
    newick = re.sub(r"^\[\&[^\]]*\]", "", newick).strip()

    stack: list[list[Tree]] = []
    current_children: list[Tree] = []
    current_nodes: list[Tree] = []
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
        current_node = Tree(name)

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


def load_newick(filepath: str | Path) -> Tree | list[Tree]:
    with open(filepath, "r") as file:
        trees = [parse_newick(newick) for newick in file]
    return trees[0] if len(trees) == 1 else trees


def to_newick(tree: Tree) -> str:
    children_newick = ",".join([to_newick(child) for child in tree.children])
    newick = tree.name
    if tree.metadata:
        reprs = {k: repr(v).replace("'", '"') for k, v in tree.metadata.items()}
        for k, r in reprs.items():
            if "," in k or "=" in k or "]" in k:
                raise ValueError(
                    f"Invalid feature key `{k}`: keys must not contain ',', '=', or ']'"
                )
            if "=" in r or "]" in r:
                raise ValueError(
                    f"Invalid value  `{r}` for feature `{k}`: values must not contain '=' or ']'"
                )
        features = [f"{k}={repr}" for k, repr in reprs.items()]
        newick += f"[&{','.join(features)}]"
    if children_newick:
        newick = f"({children_newick}){newick}"
    if tree.branch_length is not None:
        newick += f":{tree.branch_length}"
    return newick


def dump_newick(trees: Tree | list[Tree], filepath: str | Path) -> None:
    if isinstance(trees, Tree):
        trees = [trees]
    with open(filepath, "w") as file:
        for t in trees:
            file.write(to_newick(t) + ";\n")
