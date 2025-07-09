from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Generic, Literal, TypeVar

from numpy.random import Generator, default_rng
from pydantic import Field

import phylogenie.typings as pgt
from phylogenie.configs import StrictBaseModel

_T = TypeVar("_T")


class Type(str, Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    WEIBULL = "weibull"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"
    INT_UNIFORM = "int-uniform"
    CATEGORICAL = "categorical"


class Distribution(StrictBaseModel, ABC, Generic[_T]):
    @abstractmethod
    def _sample(self, rng: Generator) -> _T: ...

    def sample(self, rng: int | Generator | None = None) -> _T:
        if not isinstance(rng, Generator):
            rng = default_rng(rng)
        return self._sample(rng)


class Scalar(Distribution[pgt.Scalar]): ...


class Uniform(Scalar):
    type: Literal[Type.UNIFORM] = Type.UNIFORM
    low: float
    high: float

    def _sample(self, rng: Generator) -> float:
        return rng.uniform(self.low, self.high)


class Normal(Scalar):
    type: Literal[Type.NORMAL] = Type.NORMAL
    mean: float
    std: float

    def _sample(self, rng: Generator) -> float:
        return rng.normal(self.mean, self.std)


class LogNormal(Scalar):
    type: Literal[Type.LOGNORMAL] = Type.LOGNORMAL
    mean: float
    std: float

    def _sample(self, rng: Generator) -> float:
        return rng.lognormal(self.mean, self.std)


class Weibull(Scalar):
    type: Literal[Type.WEIBULL] = Type.WEIBULL
    scale: float
    shape: float

    def _sample(self, rng: Generator) -> float:
        return rng.weibull(self.shape) * self.scale


class Exponential(Scalar):
    type: Literal[Type.EXPONENTIAL] = Type.EXPONENTIAL
    scale: float

    def _sample(self, rng: Generator) -> float:
        return rng.exponential(scale=self.scale)


class Gamma(Scalar):
    type: Literal[Type.GAMMA] = Type.GAMMA
    scale: float
    shape: float

    def _sample(self, rng: Generator) -> float:
        return rng.gamma(shape=self.shape, scale=self.scale)


class Beta(Scalar):
    type: Literal[Type.BETA] = Type.BETA
    alpha: float
    beta: float

    def _sample(self, rng: Generator) -> float:
        return rng.beta(a=self.alpha, b=self.beta)


class IntUniform(Scalar):
    type: Literal[Type.INT_UNIFORM] = Type.INT_UNIFORM
    low: int
    high: int

    def _sample(self, rng: Generator) -> int:
        return int(rng.integers(self.low, self.high))


class Categorical(Distribution[str]):
    type: Literal[Type.CATEGORICAL] = Type.CATEGORICAL
    categories: list[str]
    probabilities: list[float]

    def _sample(self, rng: Generator) -> str:
        return str(rng.choice(self.categories, p=self.probabilities))


ScalarDistributionConfig = Annotated[
    Uniform | Normal | LogNormal | Weibull | Exponential | Gamma | Beta | IntUniform,
    Field(discriminator="type"),
]
DistributionConfig = Annotated[
    ScalarDistributionConfig | Categorical, Field(discriminator="type")
]
