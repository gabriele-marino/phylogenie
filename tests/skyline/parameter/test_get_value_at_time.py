import pytest

from phylogenie.skyline import SkylineParameter


@pytest.fixture
def param():
    return SkylineParameter([5, 2, 3], [1.0, 2.0])


def test_get_value_before_first_change(param: SkylineParameter):
    assert param.get_value_at_time(0.5) == 5


def test_get_value_exactly_at_change_time(param: SkylineParameter):
    assert param.get_value_at_time(1.0) == 2
    assert param.get_value_at_time(2.0) == 3


def test_get_value_between_changes(param: SkylineParameter):
    assert param.get_value_at_time(1.5) == 2


def test_get_value_after_last_change(param: SkylineParameter):
    assert param.get_value_at_time(3.0) == 3
    assert param.get_value_at_time(100) == 3


def test_get_value_constant_parameter():
    sp = SkylineParameter(7)
    assert sp.get_value_at_time(0) == 7
    assert sp.get_value_at_time(10) == 7


def test_with_invalid_time(param: SkylineParameter):
    with pytest.raises(ValueError):
        param.get_value_at_time(-1)
