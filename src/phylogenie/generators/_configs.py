from enum import Enum
from typing import Annotated, Any, Literal

from numpy.random import Generator
from pydantic import BaseModel, ConfigDict, Field

import phylogenie._typings as pgt


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Distribution(BaseModel):
    type: str
    model_config = ConfigDict(extra="allow")

    @property
    def args(self) -> dict[str, Any]:
        assert self.model_extra is not None
        return self.model_extra

    def __call__(self, rng: Generator) -> Any:
        return getattr(rng, self.type)(**self.args)


Integer = str | int
Scalar = str | pgt.Scalar
ManyScalars = str | pgt.Many[Scalar]
OneOrManyScalars = Scalar | pgt.Many[Scalar]
OneOrMany2DScalars = Scalar | pgt.Many2D[Scalar]


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


class TimedEventType(str, Enum):
    SAMPLING = "sampling"


class TimedEventModel(StrictBaseModel):
    times: ManyScalars


class TimedSamplingModel(TimedEventModel):
    type: Literal[TimedEventType.SAMPLING]
    state: str
    proportion: Scalar
    removal: bool


UnboundedPopulationTimedEvent = Annotated[
    TimedSamplingModel, Field(discriminator="type")
]
