import re
import subprocess
from collections.abc import Iterable
from xml.dom import minidom
from xml.etree.ElementTree import Element, tostring

from phylogenie.backend.remaster.reactions import (
    DEFAULT_POPULATION,
    SAMPLE_POPULATION,
    PunctualReaction,
    Reaction,
)
from phylogenie.skyline import skyline_parameter
from phylogenie.utils import vectorify1D

TREE_ID = "Tree"


def _beautify_xml(xml: bytes) -> str:
    xml_str = minidom.parseString(xml).toprettyxml(indent="\t")
    xml_str = re.sub(r"\n\s*\n+", "\n", xml_str)
    xml_str = xml_str.strip()
    return xml_str


def _generate_config_file(
    output_xml_file: str,
    tree_file_name: str,
    populations: list[str],
    init_values: list[int],
    sample_population: str,
    reactions: Iterable[Reaction] | None = None,
    punctual_reactions: Iterable[PunctualReaction] | None = None,
    trajectory_attrs: dict[str, str] | None = None,
    n_simulations: int = 1,
) -> None:
    simulate = Element("simulate", {"spec": "SimulatedTree", "id": TREE_ID})

    if trajectory_attrs is None:
        trajectory_attrs = {}
    trajectory = Element(
        "trajectory", {"spec": "StochasticTrajectory", **trajectory_attrs}
    )

    for population, init_value in zip(populations, init_values):
        trajectory.append(
            Element(
                "population",
                {"spec": "RealParameter", "id": population, "value": str(init_value)},
            )
        )
    trajectory.append(
        Element(
            "samplePopulation",
            {"spec": "RealParameter", "id": sample_population, "value": "0"},
        )
    )

    if reactions is not None:
        for reaction in reactions:
            if not reaction.rate:
                continue
            rate = skyline_parameter(reaction.rate)
            attrs = {
                "spec": "Reaction",
                "rate": " ".join(map(str, rate.value)),
                "value": reaction.value,
            }
            if rate.change_times:
                attrs["changeTimes"] = " ".join(map(str, rate.change_times))
            trajectory.append(Element("reaction", attrs))

    if punctual_reactions is not None:
        for punctual_reaction in punctual_reactions:
            attrs = {
                "spec": "PunctualReaction",
                "value": punctual_reaction.value,
                "times": " ".join(map(str, vectorify1D(punctual_reaction.times))),
            }
            if punctual_reaction.p is not None:
                attrs["p"] = " ".join(map(str, vectorify1D(punctual_reaction.p)))
            if punctual_reaction.n is not None:
                attrs["n"] = " ".join(map(str, vectorify1D(punctual_reaction.n)))
            trajectory.append(Element("reaction", attrs))

    simulate.append(trajectory)

    logger = Element(
        "logger", {"spec": "Logger", "mode": "tree", "fileName": tree_file_name}
    )
    logger.append(
        Element(
            "log",
            {
                "spec": "TypedTreeLogger",
                "tree": f"@{TREE_ID}",
                "removeSingletonNodes": "true",
                "noLabels": "true",
            },
        )
    )

    run = Element("run", {"spec": "Simulator", "nSims": str(n_simulations)})
    run.append(simulate)
    run.append(logger)

    beast = Element(
        "beast",
        {
            "version": "2.0",
            "namespace": ":".join(
                ["beast.base.inference", "beast.base.inference.parameter", "remaster"]
            ),
        },
    )
    beast.append(run)

    with open(output_xml_file, "w") as f:
        f.write(_beautify_xml(tostring(beast, method="xml")))


def _postprocess_tree(input_file: str, output_file: str, attributes: list[str]) -> None:

    def _replace_metadata(match: re.Match[str]) -> str:
        metadata = match.group(0)
        attrs: list[tuple[str, str]] = re.findall(r'(\w+)=(".*?"|[^,)\]]+)', metadata)
        values = [v.strip('"') for k, v in attrs if k in attributes]
        return "|" + "|".join(values)

    with open(input_file, "r") as infile:
        with open(output_file, "w") as outfile:
            for line in infile:
                line = line.strip()
                if line.lower().startswith("tree"):
                    parts = line.split("=", 1)
                    newick = parts[1].strip()
                    transformed_newick = re.sub(
                        r"\[\&[^\]]*\]", _replace_metadata, newick
                    )
                    outfile.write(transformed_newick + "\n")


def generate_trees(
    tree_file_name: str,
    populations: str | list[str] = DEFAULT_POPULATION,
    init_population: str = DEFAULT_POPULATION,
    sample_population: str = SAMPLE_POPULATION,
    reactions: Iterable[Reaction] | None = None,
    punctual_reactions: Iterable[PunctualReaction] | None = None,
    trajectory_attrs: dict[str, str] | None = None,
    output_xml_file: str | None = None,
    n_simulations: int = 1,
    seed: int | None = None,
) -> None:
    if isinstance(populations, str):
        populations = [populations]
    init_values = [0] * len(populations)
    init_values[populations.index(init_population)] = 1

    if output_xml_file is None:
        xml_file = f"{tree_file_name}-temp.xml"
    else:
        xml_file = output_xml_file

    temp_tree_file = f"{tree_file_name}-temp.nex"
    _generate_config_file(
        output_xml_file=xml_file,
        tree_file_name=temp_tree_file,
        populations=populations,
        init_values=init_values,
        sample_population=sample_population,
        reactions=reactions,
        punctual_reactions=punctual_reactions,
        trajectory_attrs=trajectory_attrs,
        n_simulations=n_simulations,
    )

    cmd = ["beast"]
    if seed is not None:
        cmd.extend(["-seed", str(seed)])
    cmd.append(xml_file)
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    _postprocess_tree(temp_tree_file, tree_file_name, ["type", "time"])
    if output_xml_file is None:
        subprocess.run(["rm", xml_file], check=True)
    subprocess.run(["rm", temp_tree_file], check=True)
