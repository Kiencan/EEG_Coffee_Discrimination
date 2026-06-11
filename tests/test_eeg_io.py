import numpy as np
from src.eeg_io import find_code_runs, label_for_code

def test_find_code_runs_basic():
    codes = np.array([0, 0, 1, 1, 1, 0, 981, 981, 981])
    runs = find_code_runs(codes)
    assert runs == [(0, 0, 2), (1, 2, 3), (0, 5, 1), (981, 6, 3)]

def test_find_code_runs_empty():
    assert find_code_runs(np.array([])) == []

def test_label_for_code():
    assert label_for_code(981) == "Arabica"
    assert label_for_code(585) == "Robusta"
    assert label_for_code(712) is None   # control
    assert label_for_code(0) is None
