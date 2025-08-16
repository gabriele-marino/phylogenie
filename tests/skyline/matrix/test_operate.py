import pytest

from phylogenie.skyline import SkylineMatrix, SkylineParameter, SkylineVector


@pytest.fixture
def scalar():
    return 10


@pytest.fixture
def sp():
    return SkylineParameter([5, 2], [4.0])


@pytest.fixture
def sv():
    return SkylineVector(value=[[3, 4], [4, 5]], change_times=[1.0])


@pytest.fixture
def sm1():
    return SkylineMatrix(value=[[[4, 2], [3, 1]], [[1, 3], [2, 4]]], change_times=[2.0])


@pytest.fixture
def sm2():
    return SkylineMatrix(value=[[[2, 5], [1, 3]], [[2, 4], [1, 3]]], change_times=[1.0])


def test_addition(
    scalar: int,
    sp: SkylineParameter,
    sv: SkylineVector,
    sm1: SkylineMatrix,
    sm2: SkylineMatrix,
):
    assert sm1 + scalar == SkylineMatrix(
        value=[[[14, 12], [13, 11]], [[11, 13], [12, 14]]], change_times=[2.0]
    )
    assert scalar + sm1 == SkylineMatrix(
        value=[[[14, 12], [13, 11]], [[11, 13], [12, 14]]], change_times=[2.0]
    )
    assert sp + sm1 == SkylineMatrix(
        value=[[[9, 7], [8, 6]], [[6, 8], [7, 9]], [[3, 5], [4, 6]]],
        change_times=[2.0, 4.0],
    )
    assert sm1 + sp == SkylineMatrix(
        value=[[[9, 7], [8, 6]], [[6, 8], [7, 9]], [[3, 5], [4, 6]]],
        change_times=[2.0, 4.0],
    )
    assert sv + sm1 == SkylineMatrix(
        value=[[[7, 5], [7, 5]], [[8, 6], [8, 6]], [[5, 7], [7, 9]]],
        change_times=[1.0, 2.0],
    )
    assert sm1 + sv == SkylineMatrix(
        value=[[[7, 5], [7, 5]], [[8, 6], [8, 6]], [[5, 7], [7, 9]]],
        change_times=[1.0, 2.0],
    )
    assert sm1 + sm2 == SkylineMatrix(
        value=[[[6, 7], [4, 4]], [[6, 6], [4, 4]], [[3, 7], [3, 7]]],
        change_times=[1.0, 2.0],
    )


def test_subtraciton(
    scalar: int,
    sp: SkylineParameter,
    sv: SkylineVector,
    sm1: SkylineMatrix,
    sm2: SkylineMatrix,
):
    assert sm1 - scalar == SkylineMatrix(
        value=[[[-6, -8], [-7, -9]], [[-9, -7], [-8, -6]]], change_times=[2.0]
    )
    assert scalar - sm1 == SkylineMatrix(
        value=[[[6, 8], [7, 9]], [[9, 7], [8, 6]]], change_times=[2.0]
    )
    assert sp - sm1 == SkylineMatrix(
        value=[[[1, 3], [2, 4]], [[4, 2], [3, 1]], [[1, -1], [0, -2]]],
        change_times=[2.0, 4.0],
    )
    assert sm1 - sp == SkylineMatrix(
        value=[[[-1, -3], [-2, -4]], [[-4, -2], [-3, -1]], [[-1, 1], [0, 2]]],
        change_times=[2.0, 4.0],
    )
    assert sv - sm1 == SkylineMatrix(
        value=[[[-1, 1], [1, 3]], [[0, 2], [2, 4]], [[3, 1], [3, 1]]],
        change_times=[1.0, 2.0],
    )
    assert sm1 - sv == SkylineMatrix(
        value=[[[1, -1], [-1, -3]], [[0, -2], [-2, -4]], [[-3, -1], [-3, -1]]],
        change_times=[1.0, 2.0],
    )
    assert sm1 - sm2 == SkylineMatrix(
        value=[[[2, -3], [2, -2]], [[2, -2], [2, -2]], [[-1, -1], [1, 1]]],
        change_times=[1.0, 2.0],
    )


def test_multiplication(
    scalar: int,
    sp: SkylineParameter,
    sv: SkylineVector,
    sm1: SkylineMatrix,
    sm2: SkylineMatrix,
):
    assert sm1 * scalar == SkylineMatrix(
        value=[[[40, 20], [30, 10]], [[10, 30], [20, 40]]], change_times=[2.0]
    )
    assert scalar * sm1 == SkylineMatrix(
        value=[[[40, 20], [30, 10]], [[10, 30], [20, 40]]], change_times=[2.0]
    )
    assert sp * sm1 == SkylineMatrix(
        value=[[[20, 10], [15, 5]], [[5, 15], [10, 20]], [[2, 6], [4, 8]]],
        change_times=[2.0, 4.0],
    )
    assert sm1 * sp == SkylineMatrix(
        value=[[[20, 10], [15, 5]], [[5, 15], [10, 20]], [[2, 6], [4, 8]]],
        change_times=[2.0, 4.0],
    )
    assert sv * sm1 == SkylineMatrix(
        value=[[[12, 6], [12, 4]], [[16, 8], [15, 5]], [[4, 12], [10, 20]]],
        change_times=[1.0, 2.0],
    )
    assert sm1 * sv == SkylineMatrix(
        value=[[[12, 6], [12, 4]], [[16, 8], [15, 5]], [[4, 12], [10, 20]]],
        change_times=[1.0, 2.0],
    )
    assert sm1 * sm2 == SkylineMatrix(
        value=[[[8, 10], [3, 3]], [[8, 8], [3, 3]], [[2, 12], [2, 12]]],
        change_times=[1.0, 2.0],
    )


def test_division(
    scalar: int,
    sp: SkylineParameter,
    sv: SkylineVector,
    sm1: SkylineMatrix,
    sm2: SkylineMatrix,
):
    assert sm1 / scalar == SkylineMatrix(
        value=[[[0.4, 0.2], [0.3, 0.1]], [[0.1, 0.3], [0.2, 0.4]]], change_times=[2.0]
    )
    assert scalar / sm1 == SkylineMatrix(
        value=[[[2.5, 5], [10 / 3, 10]], [[10, 10 / 3], [5, 2.5]]], change_times=[2.0]
    )
    assert sp / sm1 == SkylineMatrix(
        value=[
            [[1.25, 2.5], [5 / 3, 5]],
            [[5, 5 / 3], [2.5, 1.25]],
            [[2, 2 / 3], [1, 0.5]],
        ],
        change_times=[2.0, 4.0],
    )
    assert sm1 / sp == SkylineMatrix(
        value=[
            [[0.8, 0.4], [0.6, 0.2]],
            [[0.2, 0.6], [0.4, 0.8]],
            [[0.5, 1.5], [1, 2]],
        ],
        change_times=[2.0, 4.0],
    )
    assert sv / sm1 == SkylineMatrix(
        value=[
            [[0.75, 1.5], [4 / 3, 4]],
            [[1, 2], [5 / 3, 5]],
            [[4, 4 / 3], [2.5, 1.25]],
        ],
        change_times=[1.0, 2.0],
    )
    assert sm1 / sv == SkylineMatrix(
        value=[
            [[4 / 3, 2 / 3], [0.75, 0.25]],
            [[1, 0.5], [0.6, 0.2]],
            [[0.25, 0.75], [0.4, 0.8]],
        ],
        change_times=[1.0, 2.0],
    )
    assert sm1 / sm2 == SkylineMatrix(
        value=[
            [[2, 0.4], [3, 1 / 3]],
            [[2, 0.5], [3, 1 / 3]],
            [[0.5, 0.75], [2, 4 / 3]],
        ],
        change_times=[1.0, 2.0],
    )


def test_operate_with_invalid_sizes():
    with pytest.raises(ValueError):
        _ = SkylineMatrix([[1, 2], [2, 3], [3, 4]]) * SkylineMatrix([[1, 2], [2, 3]])
    with pytest.raises(ValueError):
        _ = SkylineMatrix([[1, 2], [2, 3], [3, 4]]) * SkylineMatrix([[1]])
    with pytest.raises(ValueError):
        _ = SkylineMatrix([[1, 2], [2, 3], [3, 4]]) * SkylineVector([1, 2, 3, 4])
    with pytest.raises(ValueError):
        _ = SkylineMatrix([[1, 2], [2, 3], [3, 4]]) * SkylineVector([1])
