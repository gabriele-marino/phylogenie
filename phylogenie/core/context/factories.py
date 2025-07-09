from numpy.random import Generator

import phylogenie.core.context.configs as cfg
import phylogenie.typings as pgt
from phylogenie.core.context import distributions
from phylogenie.core.typings import Data


def _sample_vector1D(
    x: distributions.Scalar, N: int, rng: Generator
) -> list[pgt.Scalar]:
    return [x.sample(rng) for _ in range(N)]


def _sample_vector2D(
    x: distributions.Scalar, N: int, zero_diagonal: bool, rng: Generator
) -> list[list[pgt.Scalar]]:
    v = [_sample_vector1D(x, N, rng) for _ in range(N)]
    if zero_diagonal:
        for i in range(N):
            v[i][i] = 0
    return v


def _sample_vector3D(
    x: distributions.Scalar, N: int, T: int, zero_diagonal: bool, rng: Generator
) -> list[list[list[pgt.Scalar]]]:
    return [_sample_vector2D(x, N, zero_diagonal, rng) for _ in range(T)]


def context_factory(x: cfg.ContextConfig, rng: Generator) -> Data:
    data: Data = {}
    for key, value in x.items():
        if isinstance(value, distributions.Distribution):
            data[key] = value.sample(rng)
        elif isinstance(value, cfg.Vector1DModel):
            data[key] = _sample_vector1D(value.x, value.N, rng)
        elif isinstance(value, cfg.Vector2DModel):
            data[key] = _sample_vector2D(value.x, value.N, value.zero_diagonal, rng)
        else:
            data[key] = _sample_vector3D(
                value.x, value.N, value.T, value.zero_diagonal, rng
            )
    return data
