import pytest

from phylogenie.skyline import SkylineMatrix


@pytest.fixture
def matrix():
    return SkylineMatrix(
        value=[[[3, 2], [4, 3]], [[1, 2], [3, 2]], [[0, 0], [0, 0]]],
        change_times=[1.0, 2.0],
    )


def test_get_value_at_time_before_first_change(matrix: SkylineMatrix):
    assert matrix.get_value_at_time(0.5) == ((3, 2), (4, 3))


def test_get_value_at_time_exact_change(matrix: SkylineMatrix):
    assert matrix.get_value_at_time(1.0) == ((1, 2), (3, 2))
    assert matrix.get_value_at_time(2.0) == ((0, 0), (0, 0))


def test_get_value_at_time_between_changes(matrix: SkylineMatrix):
    assert matrix.get_value_at_time(1.5) == ((1, 2), (3, 2))


def test_get_value_at_time_after_last_change(matrix: SkylineMatrix):
    assert matrix.get_value_at_time(3.0) == ((0, 0), (0, 0))
    assert matrix.get_value_at_time(100) == ((0, 0), (0, 0))


def test_get_value_at_time_constant_parameter():
    sv = SkylineMatrix([[7]])
    assert sv.get_value_at_time(0) == ((7,),)
    assert sv.get_value_at_time(10) == ((7,),)


def test_get_value_at_time_with_invalid_time(matrix: SkylineMatrix):
    with pytest.raises(ValueError):
        matrix.get_value_at_time(-1)
