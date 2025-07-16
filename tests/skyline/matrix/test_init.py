import pytest

from phylogenie.skyline import SkylineMatrix, SkylineParameter, SkylineVector


def test_init_with_params():
    sv1 = SkylineVector(value=[[5, 2], [6, 7], [5, 2]], change_times=[1.0, 4.0])
    sp = SkylineParameter(value=[5, 6], change_times=[2.0])
    sv2 = [sp, 1]
    sm = SkylineMatrix([sv1, sv2])

    assert sm.N == 2
    assert sm.params[0] == sv1
    assert sm.params[1] == SkylineVector([sp, 1])
    assert sm.value == [
        [[5, 2], [5, 1]],
        [[6, 7], [5, 1]],
        [[6, 7], [6, 1]],
        [[5, 2], [6, 1]],
    ]
    assert sm.change_times == [1.0, 2.0, 4.0]


def test_init_with_value_and_change_times():
    sv = SkylineMatrix(value=[[[5, 2], [3, 4]], [[6, 7], [8, 9]]], change_times=[1.0])
    assert sv.N == 2
    assert sv.params[0] == SkylineVector(value=[[5, 2], [6, 7]], change_times=[1.0])
    assert sv.params[1] == SkylineVector(value=[[3, 4], [8, 9]], change_times=[1.0])
    assert sv.value == [[[5, 2], [3, 4]], [[6, 7], [8, 9]]]
    assert sv.change_times == [1.0]


def test_init_with_mismatched_lengths():
    with pytest.raises(ValueError):
        SkylineMatrix(
            value=[[[5, 2], [3, 4]], [[6, 7], [8, 9]]], change_times=[1.0, 2.0]
        )


def test_init_with_unsorted_change_times():
    with pytest.raises(ValueError):
        SkylineMatrix(
            value=[[[5, 2], [3, 4]], [[6, 7], [8, 9]]], change_times=[2.0, 1.0]
        )


def test_init_with_negative_change_times():
    with pytest.raises(ValueError):
        SkylineMatrix(value=[[[5, 2], [3, 4]], [[6, 7], [8, 9]]], change_times=[-1.0])


def test_init_with_value_only():
    with pytest.raises(ValueError):
        SkylineMatrix(value=[[[5, 2], [3, 4]], [[6, 7], [8, 9]]])


def test_init_with_mismatched_value_dimensions():
    with pytest.raises(ValueError):
        SkylineMatrix(
            value=[[[5, 2], [3, 4]], [[6, 7, 6], [8, 9, 6], [1, 2, 3]]],
            change_times=[1.0],
        )


def test_init_with_both_params_and_value():
    with pytest.raises(ValueError):
        SkylineMatrix(params=[[5, 6], [6, 7]], value=[[[5]], [[2]]], change_times=[1.0])


def test_init_with_invalid_types():
    with pytest.raises(TypeError):
        SkylineMatrix(SkylineParameter(5))  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineMatrix(5)  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineMatrix(SkylineVector([5, 6]))  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineMatrix([[SkylineParameter(5), "a"], [1, 2]])  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineMatrix([["a", "b"], ["c", "d"]])  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineMatrix(value=[[["a"]], [["b"]]], change_times=[1.0])  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineMatrix(value=[[[5]], [[2]]], change_times="a")  # pyright: ignore
