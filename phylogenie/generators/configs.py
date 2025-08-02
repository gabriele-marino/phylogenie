from pydantic import BaseModel, ConfigDict

import phylogenie.typings as pgt


class Distribution(BaseModel):
    type: str
    model_config = ConfigDict(extra="allow")


Integer = str | int
Scalar = str | pgt.Scalar
ManyScalars = str | list[Scalar]
OneOrManyScalars = Scalar | list[Scalar]
OneOrMany2DScalars = Scalar | list[list[Scalar]]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SkylineParameterModel(StrictBaseModel):
    value: ManyScalars
    change_times: ManyScalars


class SkylineVectorModel(StrictBaseModel):
    value: str | list[OneOrManyScalars]
    change_times: ManyScalars


class SkylineMatrixModel(StrictBaseModel):
    value: str | list[OneOrMany2DScalars]
    change_times: ManyScalars


SkylineParameter = Scalar | SkylineParameterModel
SkylineVector = str | pgt.Scalar | list[SkylineParameter] | SkylineVectorModel
SkylineMatrix = str | pgt.Scalar | list[SkylineVector] | SkylineMatrixModel | None
