import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.calcoli import somma

def test_somma():
    assert somma(3, 5) == 8
    assert somma(-1, 1) == 0
    assert somma(0, 0) == 0
    assert somma(-3, -7) == -10