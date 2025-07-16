from numpy.random import Generator

import phylogenie.core.context.configs as cfg
import phylogenie.typings as pgt
from phylogenie.core.context import distributions


def _sample_vector1D(x: distributions.Scalar, N: int, rng: Generator) -> pgt.Vector1D:
    return [x.sample(rng) for _ in range(N)]


def _sample_vector2D(
    x: distributions.Scalar,
    size: tuple[int, int],
    zero_diagonal: bool,
    rng: Generator,
) -> pgt.Vector2D:
    n_rows, n_cols = size
    v = [_sample_vector1D(x, n_cols, rng) for _ in range(n_rows)]
    if zero_diagonal:
        if n_rows != n_cols:
            raise ValueError(
                f"It is impossible to initialize a non-square matrix with zero the diagonal (got x={x}, size={size} and zero_diagonal=True)"
            )
        for i in range(n_rows):
            v[i][i] = 0
    return v


def _sample_vector3D(
    x: distributions.Scalar,
    size: tuple[int, int, int],
    zero_diagonal: bool,
    rng: Generator,
) -> pgt.Vector3D:
    n_matrices, n_rows, n_cols = size
    return [
        _sample_vector2D(x, (n_rows, n_cols), zero_diagonal, rng)
        for _ in range(n_matrices)
    ]


def context_factory(x: cfg.ContextConfig, rng: Generator) -> pgt.Data:
    data: pgt.Data = {}
    for key, value in x.items():
        if isinstance(value, distributions.Distribution):
            data[key] = value.sample(rng)
        elif isinstance(value, cfg.Vector1DModel):
            data[key] = _sample_vector1D(value.x, value.size, rng)
        elif isinstance(value, cfg.Vector2DModel):
            data[key] = _sample_vector2D(value.x, value.size, value.zero_diagonal, rng)
        else:
            data[key] = _sample_vector3D(value.x, value.size, value.zero_diagonal, rng)
    return data
