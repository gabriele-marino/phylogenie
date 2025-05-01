import os
from argparse import ArgumentParser
from glob import glob

from phylogenie.generators import DatasetGenerator


def main() -> None:
    parser = ArgumentParser(
        description="Generate tree dataset(s) starting from provided config(s)."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to a config file or a directory containing config files.",
    )
    args = parser.parse_args()
    config_path = args.config_path

    if os.path.isdir(config_path):
        for config_file in glob(os.path.join(config_path, "**/*.yaml"), recursive=True):
            DatasetGenerator.from_yaml(config_file).run()
    else:
        DatasetGenerator.from_yaml(config_path).run()


if __name__ == "__main__":
    main()
