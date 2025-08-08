import pytest

from phylogenie.skyline import (
    SkylineMatrix,
    SkylineParameter,
    SkylineVector,
    skyline_matrix,
)


def test_constructor_with_skyline_parameter_like():
    sp = SkylineParameter([5, 2], [1.0])
    assert skyline_matrix(sp, 3, 4) == SkylineMatrix([[sp] * 4] * 3)
    assert skyline_matrix([[sp]], 1, 1) == SkylineMatrix([[sp]])
    assert skyline_matrix(10, 4, 4) == SkylineMatrix([[10] * 4] * 4)


def test_constructor_with_skyline_vector_like():
    sp = SkylineParameter([3, 4], [1.0])
    assert skyline_matrix([sp, 10], 2, 3) == SkylineMatrix([[sp, sp, sp], [10, 10, 10]])
    assert skyline_matrix([sp, 10], 3, 2) == SkylineMatrix(
        [[sp, 10], [sp, 10], [sp, 10]]
    )

    sv = SkylineVector(value=[[5, 2], [1, 2]], change_times=[4.0])
    assert skyline_matrix(sv, 2, 3) == SkylineMatrix(
        [[sv[0], sv[0], sv[0]], [sv[1], sv[1], sv[1]]]
    )
    assert skyline_matrix(sv, 3, 2) == SkylineMatrix(
        [[sv[0], sv[1]], [sv[0], sv[1]], [sv[0], sv[1]]]
    )


def test_constructor_with_many_skyline_vectors_coercible():
    sv = SkylineVector(value=[[5, 2, 3, 2], [1, 2, 3, 1]], change_times=[4.0])
    sp = SkylineParameter([3, 4], [1.0])
    assert skyline_matrix([sv, [sp, 10, 20, 30], 30], 3, 4) == SkylineMatrix(
        [sv, [sp, 10, 20, 30], [30, 30, 30, 30]]
    )

    assert skyline_matrix([[0, 10, 20, 30], 10, 20], 4, 3) == SkylineMatrix(
        [[0, 10, 20], [10, 10, 20], [20, 10, 20], [30, 10, 20]]
    )


def test_constructor_with_invalid_params():
    with pytest.raises(TypeError):
        skyline_matrix("a", 1)  # pyright: ignore
    with pytest.raises(TypeError):
        skyline_matrix([["a"]], 1)  # pyright: ignore
    with pytest.raises(TypeError):
        skyline_matrix([[5, 5], [5, "a"]], 2)  # pyright: ignore
    with pytest.raises(TypeError):
        skyline_matrix([[[5, 5, 5], [5, 6, 4], [1, 2, 3]]], 2, 3)  # pyright: ignore


def test_constructor_with_invalid_shape():
    with pytest.raises(ValueError):
        skyline_matrix(5, -1, 3)
    with pytest.raises(ValueError):
        skyline_matrix(5, 0, 3)
    with pytest.raises(ValueError):
        skyline_matrix([[5, 6, 8], [2, 3, 4], [1, 2, 3]], 2, 3)
    with pytest.raises(ValueError):
        skyline_matrix([5, 6, 8], 4, 2)
