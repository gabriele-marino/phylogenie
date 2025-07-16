from phylogenie.configs import StrictBaseModel
from phylogenie.core.context.distributions import (
    DistributionConfig,
    ScalarDistributionConfig,
)


class VectorModel(StrictBaseModel):
    x: ScalarDistributionConfig


class Vector1DModel(VectorModel):
    size: int


class Vector2DModel(VectorModel):
    size: tuple[int, int]
    zero_diagonal: bool = False


class Vector3DModel(VectorModel):
    size: tuple[int, int, int]
    zero_diagonal: bool = False


ContextConfig = dict[
    str, DistributionConfig | Vector1DModel | Vector2DModel | Vector3DModel
]
