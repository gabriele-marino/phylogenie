import pytest

from phylogenie.skyline import SkylineParameter, SkylineVector


@pytest.fixture
def sv():
    return SkylineVector([SkylineParameter(5), 7])


def test_bool():
    assert bool(SkylineVector([5, 7]))
    assert not bool(SkylineVector([0, 0]))


def test_equality(sv: SkylineVector):
    assert sv == SkylineVector([SkylineParameter(5), SkylineParameter(7)])
    assert sv != SkylineVector([SkylineParameter(5), SkylineParameter(8)])
    assert sv != [5, 7]
    assert sv != [SkylineParameter(5), SkylineParameter(7)]


def test_iter(sv: SkylineVector):
    assert list(iter(sv)) == [SkylineParameter(5), SkylineParameter(7)]


def test_getitem(sv: SkylineVector):
    assert sv[0] == SkylineParameter(5)
    assert sv[1] == SkylineParameter(7)
    assert sv[0:1] == SkylineVector([5])
    assert sv[:1] == SkylineVector([5])
    assert sv[1:] == SkylineVector([7])
    assert sv[-1] == SkylineParameter(7)


def test_setitem(sv: SkylineVector):
    sv[0] = SkylineParameter(10)
    assert sv[0] == SkylineParameter(10)
    assert sv[1] == SkylineParameter(7)


def test_setitem_invalid(sv: SkylineVector):
    with pytest.raises(TypeError):
        sv[0] = "10"  # pyright: ignore
