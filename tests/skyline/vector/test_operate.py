import pytest

from phylogenie.skyline import SkylineParameter, SkylineVector


@pytest.fixture
def scalar():
    return 10


@pytest.fixture
def sp():
    return SkylineParameter([5, 2], [4.0])


@pytest.fixture
def sv1():
    return SkylineVector(value=[[3, 4], [4, 5]], change_times=[1.0])


@pytest.fixture
def sv2():
    return SkylineVector(value=[[4, 2], [3, 1]], change_times=[2.0])


def test_addition(
    scalar: int, sp: SkylineParameter, sv1: SkylineVector, sv2: SkylineVector
):
    assert sv1 + scalar == SkylineVector(value=[[13, 14], [14, 15]], change_times=[1.0])
    assert scalar + sv1 == SkylineVector(value=[[13, 14], [14, 15]], change_times=[1.0])
    assert sp + sv1 == SkylineVector(
        value=[[8, 9], [9, 10], [6, 7]], change_times=[1.0, 4.0]
    )
    assert sv1 + sp == SkylineVector(
        value=[[8, 9], [9, 10], [6, 7]], change_times=[1.0, 4.0]
    )
    assert sv1 + sv2 == SkylineVector(
        value=[[7, 6], [8, 7], [7, 6]], change_times=[1.0, 2.0]
    )


def test_subtraciton(
    scalar: int, sp: SkylineParameter, sv1: SkylineVector, sv2: SkylineVector
):
    assert sv1 - scalar == SkylineVector(value=[[-7, -6], [-6, -5]], change_times=[1.0])
    assert scalar - sv1 == SkylineVector(value=[[7, 6], [6, 5]], change_times=[1.0])
    assert sp - sv1 == SkylineVector(
        value=[[2, 1], [1, 0], [-2, -3]], change_times=[1.0, 4.0]
    )
    assert sv1 - sp == SkylineVector(
        value=[[-2, -1], [-1, 0], [2, 3]], change_times=[1.0, 4.0]
    )
    assert sv1 - sv2 == SkylineVector(
        value=[[-1, 2], [0, 3], [1, 4]], change_times=[1.0, 2.0]
    )


def test_multiplication(
    scalar: int, sp: SkylineParameter, sv1: SkylineVector, sv2: SkylineVector
):
    assert sv1 * scalar == SkylineVector(value=[[30, 40], [40, 50]], change_times=[1.0])
    assert scalar * sv1 == SkylineVector(value=[[30, 40], [40, 50]], change_times=[1.0])
    assert sp * sv1 == SkylineVector(
        value=[[15, 20], [20, 25], [8, 10]], change_times=[1.0, 4.0]
    )
    assert sv1 * sp == SkylineVector(
        value=[[15, 20], [20, 25], [8, 10]], change_times=[1.0, 4.0]
    )
    assert sv1 * sv2 == SkylineVector(
        value=[[12, 8], [16, 10], [12, 5]], change_times=[1.0, 2.0]
    )


def test_division(
    scalar: int, sp: SkylineParameter, sv1: SkylineVector, sv2: SkylineVector
):
    assert sv1 / scalar == SkylineVector(
        value=[[0.3, 0.4], [0.4, 0.5]], change_times=[1.0]
    )
    assert scalar / sv1 == SkylineVector(
        value=[[10 / 3, 2.5], [2.5, 2]], change_times=[1.0]
    )
    assert sp / sv1 == SkylineVector(
        value=[[5 / 3, 1.25], [1.25, 1], [0.5, 0.4]], change_times=[1.0, 4.0]
    )
    assert sv1 / sp == SkylineVector(
        value=[[0.6, 0.8], [0.8, 1], [2, 2.5]], change_times=[1.0, 4.0]
    )
    assert sv1 / sv2 == SkylineVector(
        value=[[0.75, 2], [1, 2.5], [4 / 3, 5]], change_times=[1.0, 2.0]
    )


def test_operate_with_invalid_sizes():
    with pytest.raises(ValueError):
        _ = SkylineVector([1, 2]) * SkylineVector([3, 4, 5])
    with pytest.raises(ValueError):
        _ = SkylineVector([1, 2]) + SkylineVector([3])
    with pytest.raises(ValueError):
        _ = SkylineVector([1, 2]) - SkylineVector([])
    with pytest.raises(ValueError):
        _ = SkylineVector([1, 2]) / SkylineVector([3, 4, 5, 6])
