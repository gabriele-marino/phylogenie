from typing import Annotated, Literal

from pydantic import Field

from phylogenie.configs import StrictBaseModel
from phylogenie.core.context.distributions import (
    DistributionConfig,
    ScalarDistributionConfig,
)


class VectorModel(StrictBaseModel):
    x: ScalarDistributionConfig
    N: int


class Vector1DModel(VectorModel):
    n_dim: Literal[1] = 1


class Vector2DModel(VectorModel):
    n_dim: Literal[2] = 2
    zero_diagonal: bool = False


class Vector3DModel(VectorModel):
    n_dim: Literal[3] = 3
    T: int
    zero_diagonal: bool = False


ContextConfig = dict[
    str,
    DistributionConfig
    | Annotated[
        Vector1DModel | Vector2DModel | Vector3DModel, Field(discriminator="n_dim")
    ],
]
