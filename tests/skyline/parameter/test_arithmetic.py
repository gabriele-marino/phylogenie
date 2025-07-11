import pytest

from phylogenie.skyline import SkylineParameter


@pytest.fixture
def scalar():
    return 10


@pytest.fixture
def sp1():
    return SkylineParameter([5, 2], [4.0])


@pytest.fixture
def sp2():
    return SkylineParameter([3, 4], [1.0])


def test_addition(scalar: int, sp1: SkylineParameter, sp2: SkylineParameter):
    assert sp1 + scalar == SkylineParameter([15, 12], [4.0])
    assert scalar + sp1 == SkylineParameter([15, 12], [4.0])
    assert sp1 + sp2 == SkylineParameter([8, 9, 6], [1.0, 4.0])


def test_subtraction(scalar: int, sp1: SkylineParameter, sp2: SkylineParameter):
    assert sp1 - scalar == SkylineParameter([-5, -8], [4.0])
    assert scalar - sp1 == SkylineParameter([5, 8], [4.0])
    assert sp1 - sp2 == SkylineParameter([2, 1, -2], [1.0, 4.0])


def test_multiplication(scalar: int, sp1: SkylineParameter, sp2: SkylineParameter):
    assert sp1 * scalar == SkylineParameter([50, 20], [4.0])
    assert scalar * sp1 == SkylineParameter([50, 20], [4.0])
    assert sp1 * sp2 == SkylineParameter([15, 20, 8], [1.0, 4.0])


def test_division(scalar: int, sp1: SkylineParameter, sp2: SkylineParameter):
    assert sp1 / scalar == SkylineParameter([0.5, 0.2], [4.0])
    assert scalar / sp1 == SkylineParameter([2, 5], [4.0])
    assert sp1 / sp2 == SkylineParameter([5 / 3, 5 / 4, 2 / 4], [1.0, 4.0])


def test_bool_true_and_false():
    assert bool(SkylineParameter(5))
    assert bool(SkylineParameter([0, 1], [3]))
    assert not SkylineParameter(0)
    assert not SkylineParameter([0, 0], [1.0])


def test_equality():
    assert SkylineParameter(5) == SkylineParameter([5, 5], [1])
    assert SkylineParameter([5, 4], [1]) == SkylineParameter([5, 4], [1])
    assert SkylineParameter([5, 4], [1]) != SkylineParameter([4, 5], [1])
    assert SkylineParameter(5) != 5
