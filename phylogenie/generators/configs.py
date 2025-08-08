from pydantic import BaseModel, ConfigDict

import phylogenie.typings as pgt


class Distribution(BaseModel):
    type: str
    model_config = ConfigDict(extra="allow")


Integer = str | int
Scalar = str | pgt.Scalar
ManyScalars = str | pgt.Many[Scalar]
OneOrManyScalars = Scalar | pgt.Many[Scalar]
OneOrMany2DScalars = Scalar | pgt.Many2D[Scalar]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SkylineParameterModel(StrictBaseModel):
    value: ManyScalars
    change_times: ManyScalars


class SkylineVectorModel(StrictBaseModel):
    value: str | pgt.Many[OneOrManyScalars]
    change_times: ManyScalars


class SkylineMatrixModel(StrictBaseModel):
    value: str | pgt.Many[OneOrMany2DScalars]
    change_times: ManyScalars


SkylineParameter = Scalar | SkylineParameterModel
SkylineVector = str | pgt.Scalar | pgt.Many[SkylineParameter] | SkylineVectorModel
SkylineMatrix = str | pgt.Scalar | pgt.Many[SkylineVector] | SkylineMatrixModel | None
