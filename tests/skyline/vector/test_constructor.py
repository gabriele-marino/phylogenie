import pytest

from phylogenie.skyline import SkylineParameter, SkylineVector, skyline_vector


def test_constructor_with_param():
    sp = SkylineParameter([5, 2], [1.0])
    assert skyline_vector(sp, 5) == SkylineVector([sp] * 5)
    assert skyline_vector([sp], 1) == SkylineVector([sp])
    assert skyline_vector(5, 4) == SkylineVector([5] * 4)


def test_constructor_with_params():
    sp1 = SkylineParameter([5, 2], [4.0])
    sp2 = SkylineParameter([3, 4], [1.0])
    scalar = 5
    assert skyline_vector([sp1, sp2, scalar], 3) == SkylineVector([sp1, sp2, scalar])


def test_constructor_with_invalid_params():
    with pytest.raises(TypeError):
        skyline_vector("a", 4)  # pyright: ignore
    with pytest.raises(TypeError):
        skyline_vector([5, "b"], 2)  # pyright: ignore


def test_constructor_with_invalid_N():
    with pytest.raises(ValueError):
        skyline_vector(5, -1)
    with pytest.raises(ValueError):
        skyline_vector(5, 0)
    with pytest.raises(ValueError):
        skyline_vector([5, 6, 8], 2)
