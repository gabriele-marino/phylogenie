from typing import Any, Sequence

from kitpy.type_hints import OneOrSequence
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema

from phylogenie.skyline.core.vector import SkylineVector
from phylogenie.skyline.rand.parameter import (
    RandomSkylineParameter,
    RandomSkylineParameterConfig,
)


class RandomSkylineVectorModel(BaseModel):
    params: OneOrSequence[RandomSkylineParameterConfig]
    broadcast: bool = True


RandomSkylineVectorConfig = (
    OneOrSequence[RandomSkylineParameterConfig] | RandomSkylineVectorModel
)


class RandomSkylineVector:
    def __init__(
        self,
        params: OneOrSequence[RandomSkylineParameterConfig],
        broadcast: bool = True,
    ):
        self.params = params
        self.broadcast = broadcast

    def sample(self, N: int | None = None) -> SkylineVector:
        if isinstance(self.params, Sequence):
            return SkylineVector(
                [
                    RandomSkylineParameter.from_config(param).sample()
                    for param in self.params
                ]
            )
        if N is None:
            raise ValueError("N must be provided to sample from singleton vectors")
        if self.broadcast:
            parameter = RandomSkylineParameter.from_config(self.params).sample()
            return SkylineVector([parameter for _ in range(N)])
        return SkylineVector(
            [RandomSkylineParameter.from_config(self.params).sample() for _ in range(N)]
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.from_config, handler(RandomSkylineVectorConfig)
        )

    @classmethod
    def from_config(cls, v: RandomSkylineVectorConfig) -> "RandomSkylineVector":
        if isinstance(v, RandomSkylineVectorModel):
            return RandomSkylineVector(v.params, v.broadcast)
        return RandomSkylineVector(v)
