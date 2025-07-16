import pytest

from phylogenie.skyline import (
    SkylineMatrix,
    SkylineParameter,
    SkylineVector,
    skyline_matrix,
)


def test_constructor_with_skyline_parameter_like():
    sp = SkylineParameter([5, 2], [1.0])
    assert skyline_matrix(sp, 5) == SkylineMatrix([[sp] * 5] * 5)
    assert skyline_matrix([[sp]], 1) == SkylineMatrix([[sp]])

    assert skyline_matrix(10, 4) == SkylineMatrix([[10] * 4] * 4)
    assert skyline_matrix(10, 3, True) == SkylineMatrix(
        [[0, 10, 10], [10, 0, 10], [10, 10, 0]]
    )


def test_constructor_with_skyline_vector_like():
    sp = SkylineParameter([3, 4], [1.0])
    assert skyline_matrix([sp, 10], 2) == SkylineMatrix([[sp, sp], [10, 10]])
    assert skyline_matrix([sp, 10], 2, True) == SkylineMatrix([[0, sp], [10, 0]])

    sv = SkylineVector(value=[[5, 2], [1, 2]], change_times=[4.0])
    assert skyline_matrix(sv, 2) == SkylineMatrix([[sv[0], sv[0]], [sv[1], sv[1]]])
    assert skyline_matrix(sv, 2, True) == SkylineMatrix([[0, sv[0]], [sv[1], 0]])


def test_constructor_with_list_of_skyline_vectors_coercible():
    sv = SkylineVector(value=[[5, 2, 3], [1, 2, 3]], change_times=[4.0])
    sp = SkylineParameter([3, 4], [1.0])
    assert skyline_matrix([sv, [sp, 10, 20], 30], 3) == SkylineMatrix(
        [sv, [sp, 10, 20], [30, 30, 30]]
    )

    assert skyline_matrix([[0, 10, 20], 10, 20], 3, True) == SkylineMatrix(
        [[0, 10, 20], [10, 0, 10], [20, 20, 0]]
    )


def test_constructor_with_invalid_params():
    with pytest.raises(TypeError):
        skyline_matrix("a", 1)  # pyright: ignore
    with pytest.raises(TypeError):
        skyline_matrix([["a"]], 1)  # pyright: ignore
    with pytest.raises(TypeError):
        skyline_matrix([[5, 5], [5, "a"]], 2)  # pyright: ignore


def test_constructor_with_invalid_N():
    with pytest.raises(ValueError):
        skyline_matrix(5, -1)
    with pytest.raises(ValueError):
        skyline_matrix(5, 0)
    with pytest.raises(ValueError):
        skyline_matrix([[5, 6, 8], [2, 3, 4], [1, 2, 3]], 2)


def test_constructor_with_invalid_diagonal():
    with pytest.raises(ValueError):
        skyline_matrix([[5, 6, 8], [2, 3, 4], [1, 2, 3]], 3, zero_diagonal=True)
