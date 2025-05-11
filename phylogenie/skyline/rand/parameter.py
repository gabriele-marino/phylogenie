from typing import Any

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema

from phylogenie.rand import RandomVariable, RandomVector, sample_vector
from phylogenie.skyline.core.parameter import SkylineParameter


class RandomSkylineParameterModel(BaseModel):
    value: RandomVector
    change_times: RandomVector | None = None


RandomSkylineParameterConfig = RandomVariable | RandomSkylineParameterModel


class RandomSkylineParameter:
    def __init__(
        self,
        value: RandomVector,
        change_times: RandomVector | None = None,
    ):
        self.value = value
        self.change_times = change_times

    def sample(self) -> SkylineParameter:
        return SkylineParameter(
            sample_vector(self.value), sample_vector(self.change_times)
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.no_info_after_validator_function(
                    cls.from_config, handler(RandomSkylineParameterConfig)
                ),
            ]
        )

    @classmethod
    def from_config(cls, p: RandomSkylineParameterConfig) -> "RandomSkylineParameter":
        if isinstance(p, RandomSkylineParameterModel):
            return RandomSkylineParameter(p.value, p.change_times)
        return RandomSkylineParameter(p)
