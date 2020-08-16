from pytest import approx
from pyballistics import get_powder_names, get_options_sample, get_full_options

def test_get_options_sample():
    res = get_options_sample()
    assert res is not None

def test_full_opts():
    res = get_full_options(get_options_sample())
    assert res is not None

def test_get_powder_names():
    res = get_powder_names()
    assert res