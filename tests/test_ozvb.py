from pytest import approx
from pyballistics import ozvb_termo, get_options_sample, ozvb_lagrange



def test_ozvb_termo():
    opts = get_options_sample()
    res = ozvb_termo(opts)
    assert res['stop_reason'] != 'error'

def test_ozvb_lagrange():
    opts = get_options_sample()
    res = ozvb_lagrange(opts)
    assert res['stop_reason'] != 'error'
