import phylogenie.typings as pgt
from phylogenie.configs import StrictBaseModel

IntConfig = str | int
ScalarConfig = str | pgt.Scalar
OneOrManyIntsConfig = str | int | pgt.Many[IntConfig]
OneOrManyScalarsConfig = str | pgt.Scalar | pgt.Many[ScalarConfig]


class SkylineParameterValueModel(StrictBaseModel):
    value: str | pgt.Many[ScalarConfig]
    change_times: OneOrManyScalarsConfig


SkylineParameterLikeConfig = str | pgt.Scalar | SkylineParameterValueModel


class SkylineVectorValueModel(StrictBaseModel):
    value: str | pgt.Many[OneOrManyScalarsConfig]
    change_times: OneOrManyScalarsConfig


SkylineVectorLikeConfig = (
    str | pgt.Scalar | pgt.Many[SkylineParameterLikeConfig] | SkylineVectorValueModel
)


class SkylineMatrixValueModel(StrictBaseModel):
    value: str | pgt.Many[pgt.OneOrMany2D[ScalarConfig]]
    change_times: OneOrManyScalarsConfig


SkylineMatrixLikeConfig = (
    str | pgt.Scalar | pgt.Many[SkylineVectorLikeConfig] | SkylineMatrixValueModel
)
