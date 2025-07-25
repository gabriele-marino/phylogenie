from pydantic import BaseModel, ConfigDict

import phylogenie.typings as pgt

IntConfig = str | int
ScalarConfig = str | pgt.Scalar
ManyIntsConfig = str | list[IntConfig]
ManyScalarsConfig = str | list[ScalarConfig]
OneOrManyScalarsConfig = ScalarConfig | list[ScalarConfig]
OneOrMany2DScalarsConfig = ScalarConfig | list[list[ScalarConfig]]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SkylineParameterValueModel(StrictBaseModel):
    value: ManyScalarsConfig
    change_times: ManyScalarsConfig


SkylineParameterLikeConfig = ScalarConfig | SkylineParameterValueModel


class SkylineVectorValueModel(StrictBaseModel):
    value: str | list[OneOrManyScalarsConfig]
    change_times: ManyScalarsConfig


SkylineVectorCoercibleConfig = (
    str | pgt.Scalar | list[SkylineParameterLikeConfig] | SkylineVectorValueModel
)


class SkylineMatrixValueModel(StrictBaseModel):
    value: str | list[OneOrMany2DScalarsConfig]
    change_times: ManyScalarsConfig


SkylineMatrixCoercibleConfig = (
    str | pgt.Scalar | list[SkylineVectorCoercibleConfig] | SkylineMatrixValueModel
)
