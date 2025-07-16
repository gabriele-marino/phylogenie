import pytest
from numpy.random import default_rng

import phylogenie.typeguards as tg
from phylogenie.core.context.configs import Vector1DModel, Vector2DModel, Vector3DModel
from phylogenie.core.context.distributions import Uniform
from phylogenie.core.context.factories import context_factory


def test_context_factory():
    dist = Uniform(low=0, high=5)
    scalar_config = dist
    vector1D_config = Vector1DModel(x=dist, size=3)
    vector2D_config = Vector2DModel(x=dist, size=(3, 4))
    vector2D_zero_diag_config = Vector2DModel(x=dist, size=(3, 3), zero_diagonal=True)
    vector3D_config = Vector3DModel(x=dist, size=(2, 3, 4))
    vector3D_zero_diag_config = Vector3DModel(
        x=dist, size=(2, 3, 3), zero_diagonal=True
    )
    rng = default_rng()

    data = context_factory(
        {
            "scalar": scalar_config,
            "vector1D": vector1D_config,
            "vector2D": vector2D_config,
            "vector2D_zero_diag": vector2D_zero_diag_config,
            "vector3D": vector3D_config,
            "vector3D_zero_diag": vector3D_zero_diag_config,
        },
        rng=rng,
    )

    assert isinstance(data["scalar"], float)
    assert 0 < data["scalar"] < 5

    assert tg.is_many_scalars(data["vector1D"])
    assert len(data["vector1D"]) == 3
    assert all(0 < x < 5 for x in data["vector1D"])

    v2D = data["vector2D"]
    assert tg.is_many_2D_scalars(v2D)
    assert len(v2D) == 3
    assert all(len(x) == 4 for x in v2D)
    assert all(0 < x < 5 for row in v2D for x in row)

    v2D_zero_diag = data["vector2D_zero_diag"]
    assert tg.is_many_2D_scalars(v2D_zero_diag)
    assert not any(v2D_zero_diag[i][i] for i in range(len(v2D_zero_diag)))

    v3D = data["vector3D"]
    assert tg.is_many_3D_scalars(v3D)
    assert len(v3D) == 2
    assert all(len(x) == 3 for x in v3D)
    assert all(len(row) == 4 for x in v3D for row in x)
    assert all(0 < x < 5 for matrix in v3D for row in matrix for x in row)

    v3D_zero_diag = data["vector3D_zero_diag"]
    assert tg.is_many_3D_scalars(v3D_zero_diag)
    assert not any(m[j][j] for m in v3D_zero_diag for j in range(len(m)))

    with pytest.raises(ValueError):
        context_factory(
            {"2D_rectangluar": Vector2DModel(x=dist, size=(3, 4), zero_diagonal=True)},
            rng=rng,
        )

    with pytest.raises(ValueError):
        context_factory(
            {
                "3D_rectangluar": Vector3DModel(
                    x=dist, size=(2, 3, 4), zero_diagonal=True
                )
            },
            rng=rng,
        )
