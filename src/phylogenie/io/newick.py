import re
from pathlib import Path

from phylogenie.core import Node, Tree


def parse_newick(newick: str, translations: dict[str, str] | None = None) -> Tree:
    newick = newick.strip()
    newick = re.sub(r"^\[\&[^\]]*\]", "", newick).strip()

    stack: list[list[Node]] = []
    current_children: list[Node] = []
    current_nodes: list[Node] = []
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
        current_node = Node(name)

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
            return Tree(current_node)

        i += 1


def load_newick(filepath: str | Path) -> Tree | list[Tree]:
    with open(filepath, "r") as file:
        trees = [parse_newick(newick) for newick in file]
    return trees[0] if len(trees) == 1 else trees


def to_newick(tree: Tree) -> str:
    def _to_newick(node: Node) -> str:
        children_newick = ",".join([_to_newick(child) for child in node.children])
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

    return _to_newick(tree.root)


def dump_newick(trees: Tree | list[Tree], filepath: str | Path) -> None:
    if isinstance(trees, Tree):
        trees = [trees]
    with open(filepath, "w") as file:
        for t in trees:
            file.write(to_newick(t) + ";\n")
