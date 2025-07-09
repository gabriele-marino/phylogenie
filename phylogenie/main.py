import os
from argparse import ArgumentParser
from glob import glob

from pydantic import TypeAdapter
from yaml import safe_load

from phylogenie.core import DatasetGeneratorConfig
from phylogenie.core.dataset import DatasetGenerator


def run(config_path: str) -> None:
    adapter: TypeAdapter[DatasetGenerator] = TypeAdapter(DatasetGeneratorConfig)

    if os.path.isdir(config_path):
        for config_file in glob(os.path.join(config_path, "**/*.yaml"), recursive=True):
            with open(config_file, "r") as f:
                config = safe_load(f)
            generator = adapter.validate_python(config)
            generator.generate()
    else:
        with open(config_path, "r") as f:
            config = safe_load(f)
        generator = adapter.validate_python(config)
        generator.generate()


def main() -> None:
    parser = ArgumentParser(
        description="Generate dataset(s) starting from provided config(s)."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to a config file or a directory containing config files.",
    )
    args = parser.parse_args()

    run(args.config_path)
