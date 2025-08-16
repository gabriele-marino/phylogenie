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
    assert sp1 / sp2 == SkylineParameter([5 / 3, 1.25, 0.5], [1.0, 4.0])
