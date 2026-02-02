import os
from argparse import ArgumentParser
from glob import glob

from pydantic import TypeAdapter, ValidationError
from yaml import safe_load

from phylogenie.generators import DatasetGeneratorConfig
from phylogenie.generators.dataset import DatasetGenerator


def _format_validation_error(e: ValidationError) -> str:
    """
    Format a Pydantic ValidationError for console output.

    Parameters
    -----------
    e : ValidationError
        Validation error to format.

    Returns
    --------
    str
        Human-readable formatted error lines.
    """
    formatted_errors = [
        f"- {'.'.join(str(loc) for loc in err['loc'])}: {err['msg']} ({err['type']})"
        for err in e.errors()
    ]
    return "\n".join(formatted_errors)


def _generate_from_config_file(config_file: str):
    """
    Load a configuration file, validate it, and run generation.

    Parameters
    -----------
    config_file : str
        Path to the YAML configuration file.
    """
    adapter: TypeAdapter[DatasetGenerator] = TypeAdapter(DatasetGeneratorConfig)
    with open(config_file, "r") as f:
        try:
            config = safe_load(f)
        except Exception as e:
            print(f"❌ Failed to parse {config_file}: {e}")
            exit(-1)
        try:
            generator = adapter.validate_python(config)
        except ValidationError as e:
            print("❌ Invalid configuration:")
            print(_format_validation_error(e))
            exit(-1)
    generator.generate()


def run(config_path: str):
    """
    Run dataset generation from a config file or directory.

    Parameters
    -----------
    config_path : str
        Path to a config file or a directory containing YAML files.
    """
    if os.path.isdir(config_path):
        for config_file in glob(os.path.join(config_path, "**/*.yaml"), recursive=True):
            _generate_from_config_file(config_file)
    else:
        _generate_from_config_file(config_path)


def main():
    """
    Entry point for the CLI.
    """
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


if __name__ == "__main__":
    run("tutorials/1.BD-Trees.yaml")
