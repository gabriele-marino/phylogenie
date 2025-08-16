import pytest

from phylogenie.skyline import SkylineParameter, SkylineVector


def test_init_with_params():
    sp1 = SkylineParameter([5, 2], [4.0])
    sp2 = SkylineParameter([3, 4], [1.0])
    sp3 = 5
    sv = SkylineVector([sp1, sp2, sp3])

    assert sv.N == 3
    assert sv.params[0] == sp1
    assert sv.params[1] == sp2
    assert sv.params[2] == SkylineParameter(5)
    assert sv.value == [[5, 3, 5], [5, 4, 5], [2, 4, 5]]
    assert sv.change_times == [1.0, 4.0]


def test_init_with_value_and_change_times():
    sv = SkylineVector(value=[[5, 2], [3, 4]], change_times=[1.0])
    assert sv.N == 2
    assert sv.params[0] == SkylineParameter([5, 3], [1.0])
    assert sv.params[1] == SkylineParameter([2, 4], [1.0])
    assert sv.value == [[5, 2], [3, 4]]
    assert sv.change_times == [1.0]


def test_init_with_mismatched_lengths():
    with pytest.raises(ValueError):
        SkylineVector(value=[[5, 2], [4, 2], [4, 5], [4, 2]], change_times=[1.0, 2.0])


def test_init_with_unsorted_change_times():
    with pytest.raises(ValueError):
        SkylineVector(value=[[5, 2], [3, 4]], change_times=[2.0, 1.0])


def test_init_with_negative_change_times():
    with pytest.raises(ValueError):
        SkylineVector(value=[[5, 2]], change_times=[-1.0])


def test_init_with_value_only():
    with pytest.raises(ValueError):
        SkylineVector(value=[[5, 2]])


def test_init_with_mismatched_value_dimensions():
    with pytest.raises(ValueError):
        SkylineVector(value=[[5, 2], [3]], change_times=[1.0])


def test_init_with_both_params_and_value():
    with pytest.raises(ValueError):
        SkylineVector(
            params=[SkylineParameter(5)], value=[[5, 2], [4, 3]], change_times=[1.0]
        )


def test_init_with_invalid_types():
    with pytest.raises(TypeError):
        SkylineVector(SkylineParameter(5))  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineVector(5)  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineVector([SkylineParameter(5), "a"])  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineVector(["a", "b"])  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineVector(value=[["a"], ["b"]], change_times=[1.0])  # pyright: ignore
    with pytest.raises(TypeError):
        SkylineVector(value=[[5, 2], [4, 3]], change_times="a")  # pyright: ignore
