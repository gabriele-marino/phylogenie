from typing import Any

from numpy.random import Generator
from pydantic import BaseModel, ConfigDict

import phylogenie.typings as pgt
from phylogenie.treesimulator import EventType


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


class Event(StrictBaseModel):
    state: str | None = None
    rate: SkylineParameter


class Mutation(Event):
    rate_scalers: dict[EventType, Distribution]
