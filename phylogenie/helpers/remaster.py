import re
import subprocess
from xml.etree.ElementTree import Element, tostring

from kitpy.xmls import beautify_xml

from phylogenie.parameterizations import Parameterization, Rates
from phylogenie.skyline import SkylineParameter

TREE_ID = "Tree"


def _get_reactions(rates: Rates, populations: list[str]) -> list[Element]:
    reactions = []
    for (
        p1,
        birth_rate,
        death_rate,
        sampling_rate,
        migration_rates_row,
        birth_rates_among_demes_row,
    ) in zip(
        populations,
        rates.birth_rates,
        rates.death_rates,
        rates.sampling_rates,
        rates.migration_rates,
        rates.birth_rates_among_demes,
    ):
        reaction_configs: list[tuple[SkylineParameter, str]] = [
            (birth_rate, f"{p1} -> 2{p1}"),
            (death_rate, f"{p1} -> 0"),
            (sampling_rate, f"{p1} -> sample"),
        ]

        for p2, migration_rate, birth_rate_among_demes in zip(
            [p for p in populations if p != p1],
            migration_rates_row,
            birth_rates_among_demes_row,
        ):
            reaction_configs.append((migration_rate, f"{p1} -> {p2}"))
            reaction_configs.append((birth_rate_among_demes, f"{p1} -> {p1} + {p2}"))

        for rate, reaction in reaction_configs:
            if not rate:
                continue
            attrs = {
                "spec": "Reaction",
                "rate": " ".join(str(v) for v in rate.value),
                "value": reaction,
            }
            if rate.change_times:
                attrs["changeTimes"] = " ".join(str(t) for t in rate.change_times)
            reactions.append(Element("reaction", attrs))
    return reactions


def _get_trajectory(
    parameterization: Parameterization,
    init_values: list[int],
    trajectory_attrs: dict[str, str] | None = None,
) -> Element:
    populations = parameterization.populations
    N = len(populations)
    if len(init_values) != N:
        raise ValueError(
            f"Number of initial values ({len(init_values)}) does not match number of populations ({N})."
        )

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
            "samplePopulation", {"spec": "RealParameter", "id": "sample", "value": "0"}
        )
    )
    trajectory.extend(_get_reactions(parameterization.rates, populations))
    return trajectory


def _prepare_config_file(
    parameterization: Parameterization,
    init_values: list[int],
    tree_file_name: str = "trees.nex",
    output_xml_file: str = "pymaster.xml",
    trajectory_attrs: dict[str, str] | None = None,
    n_simulations: int = 1,
) -> None:
    simulate = Element("simulate", {"spec": "SimulatedTree", "id": TREE_ID})
    simulate.append(_get_trajectory(parameterization, init_values, trajectory_attrs))

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
        f.write(beautify_xml(tostring(beast, method="xml")))


def _postprocess_tree(
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
                line = line.strip()
                if line.lower().startswith("tree"):
                    parts = line.split("=", 1)
                    newick = parts[1].strip()
                    transformed_newick = re.sub(
                        r"\[\&[^\]]*\]", _replace_metadata, newick
                    )
                    outfile.write(transformed_newick + "\n")


def generate_trees(
    parameterization: Parameterization,
    init_values: list[int],
    output_file: str,
    trajectory_attrs: dict[str, str] | None = None,
    xml_file: str | None = None,
    n_simulations: int = 1,
) -> None:
    if xml_file is None:
        output_xml_file = f"{output_file}-temp.xml"
    else:
        output_xml_file = xml_file

    temp_tree_file = f"{output_file}-temp.nex"
    _prepare_config_file(
        parameterization=parameterization,
        init_values=init_values,
        tree_file_name=temp_tree_file,
        output_xml_file=output_xml_file,
        trajectory_attrs=trajectory_attrs,
        n_simulations=n_simulations,
    )

    subprocess.run(
        ["beast", output_xml_file],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    _postprocess_tree(temp_tree_file, output_file, ["type", "time"])
    if xml_file is None:
        subprocess.run(["rm", output_xml_file], check=True)
    subprocess.run(["rm", temp_tree_file], check=True)
