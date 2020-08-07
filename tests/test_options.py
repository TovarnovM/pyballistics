from pytest import approx
from pyballistics.options import get_powder_names, get_termo_options_sample
import numpy as np

# TODO доделать тесты)

def test_get_powder_names():
    pnames = get_powder_names()
    assert len(pnames) > 0

if __name__ == "__main__":
    from pyballistics.termo import termo_ballistics
    print(termo_ballistics(get_termo_options_sample()))