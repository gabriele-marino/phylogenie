from typing import Any, Sequence

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema
from pykit.type_hints import OneOrMatrix

from phylogenie.skyline.core.matrix import SkylineMatrix
from phylogenie.skyline.rand.parameter import (
    RandomSkylineParameter,
    RandomSkylineParameterConfig,
)


class RandomSkylineMatrixModel(BaseModel):
    params: OneOrMatrix[RandomSkylineParameterConfig]
    broadcast: bool = True


RandomSkylineMatrixConfig = (
    OneOrMatrix[RandomSkylineParameterConfig] | RandomSkylineMatrixModel
)


class RandomSkylineMatrix:
    def __init__(
        self,
        params: OneOrMatrix[RandomSkylineParameterConfig],
        broadcast: bool = True,
    ):
        self.params = params
        self.broadcast = broadcast

    def sample(self, N: int | None = None) -> SkylineMatrix:
        if isinstance(self.params, Sequence):
            return SkylineMatrix(
                [
                    [
                        RandomSkylineParameter.from_config(param).sample()
                        for param in row
                    ]
                    for row in self.params
                ]
            )
        if N is None:
            raise ValueError("N must be provided to sample from singleton matrces")
        if self.broadcast:
            parameter = RandomSkylineParameter.from_config(self.params).sample()
            return SkylineMatrix([[parameter] * (N - 1)] * N)
        return SkylineMatrix(
            [
                [
                    RandomSkylineParameter.from_config(self.params).sample()
                    for _ in range(N - 1)
                ]
                for _ in range(N)
            ]
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.no_info_after_validator_function(
                    cls.from_config, handler(RandomSkylineMatrixConfig)
                ),
            ]
        )

    @classmethod
    def from_config(cls, m: RandomSkylineMatrixConfig) -> "RandomSkylineMatrix":
        if isinstance(m, RandomSkylineMatrixModel):
            return RandomSkylineMatrix(m.params, m.broadcast)
        return RandomSkylineMatrix(m)
