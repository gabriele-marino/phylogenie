"""
Core configuration models.

This module provides foundational configuration models used as building
blocks for generator configurations. Each configuration describes the
inputs required by a factory function to construct a specific component.

The models defined here standardize how scalar values, distributions,
and time-varying ("skyline") parameters are represented, enabling
flexible and composable configuration schemas across the codebase.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

import phylogenie.typings as pgt


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Distribution(BaseModel):
    type: str
    model_config = ConfigDict(extra="allow")

    @property
    def args(self) -> dict[str, Any]:
        return self.model_extra  # pyright: ignore


Context = dict[str, str | Distribution]
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
