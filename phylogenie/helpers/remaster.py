import subprocess
from xml.etree.ElementTree import Element, tostring

from kitpy.xmls import beautify_xml

from phylogenie.parameterizations import Parameterization, Rates
from phylogenie.skyline import SkylineParameter
from phylogenie.utils import extract_newick_from_nexus, process_newick_taxa_names

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

        for p2 in [p for p in populations if p != p1]:
            for migration_rate in migration_rates_row:
                reaction_configs.append((migration_rate, f"{p1} -> {p2}"))
            for birth_rate_among_demes in birth_rates_among_demes_row:
                reaction_configs.append(
                    (birth_rate_among_demes, f"{p1} -> {p1} + {p2}")
                )

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


def prepare_config_file(
    parameterization: Parameterization,
    init_values: list[int],
    output_tree_file: str = "trees.nex",
    output_xml_file: str = "pymaster.xml",
    trajectory_attrs: dict[str, str] | None = None,
    n_simulations: int = 1,
) -> None:
    simulate = Element("simulate", {"spec": "SimulatedTree", "id": TREE_ID})
    simulate.append(_get_trajectory(parameterization, init_values, trajectory_attrs))

    logger = Element(
        "logger", {"spec": "Logger", "mode": "tree", "fileName": output_tree_file}
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


def generate_tree(
    parameterization: Parameterization,
    init_values: list[int],
    output_file: str,
    trajectory_attrs: dict[str, str] | None = None,
) -> None:
    temp_nexus_file = f"{output_file}-temp.nex"
    temp_xml_file = f"{output_file}-temp.xml"
    prepare_config_file(
        parameterization=parameterization,
        init_values=init_values,
        output_tree_file=temp_nexus_file,
        output_xml_file=temp_xml_file,
        trajectory_attrs=trajectory_attrs,
    )

    subprocess.run(
        ["beast", temp_xml_file],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    extract_newick_from_nexus(temp_nexus_file, output_file)
    process_newick_taxa_names(output_file, output_file, ["type", "time"])
    subprocess.run(["rm", temp_nexus_file], check=True)
    subprocess.run(["rm", temp_xml_file], check=True)
