from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field
from pykit.type_hints import Numeric, OneOrSequence
from pykit.validators import ensure_list


class RandomVariableType(str, Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    WEIBULL = "weibull"
    EXPONENTIAL = "exponential"


class BaseRandomVariable(ABC, BaseModel):
    type: RandomVariableType

    @abstractmethod
    def sample(self) -> float: ...


class UniformRandomVariable(BaseRandomVariable):
    type: Literal[RandomVariableType.UNIFORM] = RandomVariableType.UNIFORM
    low: float
    high: float

    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)


class NormalRandomVariable(BaseRandomVariable):
    type: Literal[RandomVariableType.NORMAL] = RandomVariableType.NORMAL
    mean: float
    std: float

    def sample(self) -> float:
        return np.random.normal(self.mean, self.std)


class LogNormalRandomVariable(BaseRandomVariable):
    type: Literal[RandomVariableType.LOGNORMAL] = RandomVariableType.LOGNORMAL
    mean: float
    std: float

    def sample(self) -> float:
        return np.random.lognormal(self.mean, self.std)


class WeibullRandomVariable(BaseRandomVariable):
    type: Literal[RandomVariableType.WEIBULL] = RandomVariableType.WEIBULL
    scale: float
    shape: float

    def sample(self) -> float:
        return np.random.weibull(self.shape) * self.scale


class ExponentialRandomVariable(BaseRandomVariable):
    type: Literal[RandomVariableType.EXPONENTIAL] = RandomVariableType.EXPONENTIAL
    rate: float

    def sample(self) -> float:
        return np.random.exponential(1 / self.rate)


RandomVariable = (
    Numeric
    | Annotated[
        UniformRandomVariable
        | NormalRandomVariable
        | LogNormalRandomVariable
        | WeibullRandomVariable
        | ExponentialRandomVariable,
        Field(discriminator="type"),
    ]
)
RandomVector = OneOrSequence[RandomVariable]


def sample_variable(variable: RandomVariable) -> Numeric:
    return variable.sample() if isinstance(variable, BaseRandomVariable) else variable


def sample_vector(vector: RandomVector | None) -> list[Numeric]:
    vector = ensure_list(vector)
    return [sample_variable(v) for v in vector]
