import pytest

from phylogenie.skyline import SkylineParameter


def test_init_with_scalar_value():
    sp = SkylineParameter(5)
    assert sp.value == [5]
    assert sp.change_times == []


def test_init_with_value_and_change_times():
    sp = SkylineParameter([5, 2, 3], [1.0, 2.5])
    assert sp.value == [5, 2, 3]
    assert sp.change_times == [1.0, 2.5]


def test_init_removes_consecutive_duplicate_values():
    sp = SkylineParameter([3, 5, 5], [1.0, 2.0])
    assert sp.value == [3, 5]
    assert sp.change_times == [1.0]

    sp = SkylineParameter([5, 5, 5], [1.0, 2.0])
    assert sp.value == [5]
    assert sp.change_times == []


def test_init_with_invalid_types():
    with pytest.raises(TypeError):
        SkylineParameter("a")  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineParameter(["a", "b"])  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineParameter([5, 2], change_times="a")  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineParameter([5, 2], change_times=["a", "b"])  # pyright: ignore


def test_init_with_mismatched_lengths():
    with pytest.raises(ValueError):
        SkylineParameter([5, 2, 3], [1.0])
    with pytest.raises(ValueError):
        SkylineParameter([5, 2])


def test_with_negative_change_times():
    with pytest.raises(ValueError):
        SkylineParameter([5, 2], [-1.0])


def test_init_with_unsorted_change_times():
    with pytest.raises(ValueError):
        SkylineParameter([5, 2, 3], [2.0, 1.0])
