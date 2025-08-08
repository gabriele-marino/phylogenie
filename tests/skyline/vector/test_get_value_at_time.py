import pytest

from phylogenie.skyline import SkylineVector


@pytest.fixture
def vector():
    return SkylineVector(
        value=[[3, 2, 1], [4, 3, 2], [5, 4, 3]], change_times=[1.0, 2.0]
    )


def test_get_value_at_time_before_first_change(vector: SkylineVector):
    assert vector.get_value_at_time(0.5) == [3, 2, 1]


def test_get_value_at_time_exact_change(vector: SkylineVector):
    assert vector.get_value_at_time(1.0) == [4, 3, 2]
    assert vector.get_value_at_time(2.0) == [5, 4, 3]


def test_get_value_at_time_between_changes(vector: SkylineVector):
    assert vector.get_value_at_time(1.5) == [4, 3, 2]


def test_get_value_at_time_after_last_change(vector: SkylineVector):
    assert vector.get_value_at_time(3.0) == [5, 4, 3]
    assert vector.get_value_at_time(100) == [5, 4, 3]


def test_get_value_at_time_constant_parameter():
    sv = SkylineVector([7, 7, 7])
    assert sv.get_value_at_time(0) == [7, 7, 7]
    assert sv.get_value_at_time(10) == [7, 7, 7]


def test_get_value_at_time_with_invalid_time(vector: SkylineVector):
    with pytest.raises(ValueError):
        vector.get_value_at_time(-1)
