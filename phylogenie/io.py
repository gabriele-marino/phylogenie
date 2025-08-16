from typing import Callable

from phylogenie.msa import MSA, Sequence
from phylogenie.tree import Tree


def _parse_newick(newick: str) -> Tree:
    newick = newick.strip()
    stack: list[list[Tree]] = []
    current_children: list[Tree] = []
    current_nodes: list[Tree] = []
    i = 0
    while i < len(newick):

        def _parse_chars(stoppers: list[str]) -> str:
            nonlocal i
            chars = ""
            while newick[i] not in stoppers:
                chars += newick[i]
                i += 1
            return chars

        if newick[i] == "(":
            stack.append(current_nodes)
            current_nodes = []
        else:
            id = _parse_chars([":", ",", ")", ";", "["])
            branch_length = None
            if newick[i] == ":":
                i += 1
                branch_length = float(_parse_chars([",", ")", ";", "["]))

            current_node = Tree(id, branch_length)
            for node in current_children:
                current_node.add_child(node)
                current_children = []
            current_nodes.append(current_node)

            if newick[i] == "[":
                i += 1
                features = _parse_chars(["]"]).split(":")
                i += 1
                if features[0] != "&&NHX":
                    raise ValueError(f"Expected '&&NHX' for node features.")
                for feature in features[1:]:
                    key, value = feature.split("=", 1)
                    current_node.set(key, eval(value))

            if newick[i] == ")":
                current_children = current_nodes
                current_nodes = stack.pop()
            elif newick[i] == ";":
                return current_node

        i += 1

    raise ValueError("Newick string is invalid.")


def load_newick(filepath: str) -> Tree | list[Tree]:
    with open(filepath, "r") as file:
        trees = [_parse_newick(newick) for newick in file]
    return trees[0] if len(trees) == 1 else trees


def _to_newick(tree: Tree) -> str:
    children_newick = ",".join([_to_newick(child) for child in tree.children])
    newick = tree.id
    if children_newick:
        newick = f"({children_newick}){newick}"
    if tree.branch_length is not None:
        newick += f":{tree.branch_length}"
    if tree.features:
        reprs = {k: repr(v).replace("'", '"') for k, v in tree.features.items()}
        features = [f"{k}={repr}" for k, repr in reprs.items()]
        newick += f"[&&NHX:{':'.join(features)}]"
    return newick


def dump_newick(trees: Tree | list[Tree], filepath: str) -> None:
    if isinstance(trees, Tree):
        trees = [trees]
    with open(filepath, "w") as file:
        for t in trees:
            file.write(_to_newick(t) + ";\n")


def load_fasta(
    fasta_file: str, extract_time_from_id: Callable[[str], float] | None = None
) -> MSA:
    sequences: list[Sequence] = []
    with open(fasta_file, "r") as f:
        for line in f:
            if not line.startswith(">"):
                raise ValueError(f"Invalid FASTA format: expected '>', got '{line[0]}'")
            id = line[1:].strip()
            time = None
            if extract_time_from_id is not None:
                time = extract_time_from_id(id)
            elif "|" in id:
                try:
                    time = float(id.split("|")[-1])
                except ValueError:
                    pass
            chars = next(f).strip()
            sequences.append(Sequence(id, chars, time))
    return MSA(sequences)
