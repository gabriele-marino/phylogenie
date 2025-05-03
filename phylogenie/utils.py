import re


def extract_newick_from_nexus(input_file: str, output_file: str) -> None:
    with open(input_file, "r") as infile:
        with open(output_file, "w") as outfile:
            for line in infile:
                line = line.strip()
                if line.lower().startswith("tree"):
                    parts = line.split("=", 1)
                    newick = parts[1].strip()
                    outfile.write(newick + "\n")


def process_newick_taxa_names(
    input_file: str, output_file: str, attributes: list[str], sep: str = "|"
) -> None:

    def _replace_metadata(match: re.Match[str]) -> str:
        metadata = match.group(0)
        attrs = re.findall(r'(\w+)=(".*?"|[^,)\]]+)', metadata)
        values = [v.strip('"') for k, v in attrs if k in attributes]
        return sep + sep.join(values)

    with open(input_file, "r") as infile:
        with open(output_file, "w") as outfile:
            for line in infile:
                tree = line.strip()
                transformed_tree = re.sub(r"\[\&[^\]]*\]", _replace_metadata, tree)
                outfile.write(transformed_tree + "\n")
