import pytest

from phylogenie.skyline import SkylineMatrix, SkylineParameter, SkylineVector


@pytest.fixture
def sm():
    return SkylineMatrix([SkylineVector([1, 2]), [SkylineParameter(5), 7]])


def test_bool():
    assert bool(SkylineMatrix([[5]]))
    assert not bool(SkylineMatrix([[0, 0], [0, 0]]))


def test_equality(sm: SkylineMatrix):
    assert sm == SkylineMatrix([SkylineVector([1, 2]), [SkylineParameter(5), 7]])
    assert sm == SkylineMatrix([[1, 2], [5, 7]])
    assert sm != [[1, 2], [5, 7]]
    assert sm != [SkylineVector([1, 2]), SkylineVector([5, 7])]


def test_iter(sm: SkylineMatrix):
    assert list(iter(sm)) == [SkylineVector([1, 2]), SkylineVector([5, 7])]


def test_getitem(sm: SkylineMatrix):
    assert sm[0] == SkylineVector([1, 2])
    assert sm[1] == SkylineVector([5, 7])
    assert sm[0:1] == SkylineMatrix([SkylineVector([1, 2])])
    assert sm[:1] == SkylineMatrix([SkylineVector([1, 2])])
    assert sm[1:] == SkylineMatrix([SkylineVector([5, 7])])
    assert sm[-1] == SkylineVector([5, 7])
    assert sm[0, 0] == SkylineParameter(1)
    assert sm[1, 0] == SkylineParameter(5)
    assert sm[:, 0] == SkylineVector([1, 5])
    assert sm[-1, :] == SkylineVector([5, 7])


def test_setitem(sm: SkylineMatrix):
    sm[0] = SkylineVector([1, 2])
    assert sm[0] == SkylineVector([1, 2])
    assert sm[1] == SkylineVector([5, 7])


def test_setitem_invalid(sm: SkylineMatrix):
    with pytest.raises(TypeError):
        sm[0] = ["a"]  # pyright: ignore
    with pytest.raises(ValueError):
        sm[0] = [1, 2, 3]  # pyright: ignore
