from typing import Annotated

from pydantic import Field

from phylogenie.core.msas.alisim import AliSimGenerator

MSAsGeneratorConfig = Annotated[
    AliSimGenerator,
    Field(discriminator="backend"),
]
