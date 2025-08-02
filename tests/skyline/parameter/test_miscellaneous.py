from phylogenie.skyline import SkylineParameter, skyline_parameter


def test_bool():
    assert bool(SkylineParameter(5))
    assert bool(SkylineParameter([0, 1], [3]))
    assert not SkylineParameter(0)
    assert not SkylineParameter([0, 0], [1.0])


def test_equality():
    assert SkylineParameter(5) == SkylineParameter([5, 5], [1])
    assert SkylineParameter([5, 4], [1]) == SkylineParameter([5, 4], [1])
    assert SkylineParameter([5, 4], [1]) != SkylineParameter([4, 5], [1])
    assert SkylineParameter(5) != 5


def test_constructor():
    assert skyline_parameter(5) == SkylineParameter(5)
    assert skyline_parameter(SkylineParameter(5)) == SkylineParameter(5)
