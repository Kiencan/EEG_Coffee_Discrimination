import numpy as np
import pandas as pd
import pytest
from src.sensory import load_sensory_long, align_ratings
from src.config import ROOT


def test_load_sensory_long_real_file():
    long = load_sensory_long(ROOT / "protocol" / "sensory_data.csv")
    for col in ["subject", "trial", "code", "valence", "intensity", "favourite"]:
        assert col in long.columns
    assert long["subject"].nunique() == 20
    assert (long.groupby("subject").size() == 45).all()
    assert long["valence"].between(1, 7).all()
    assert long["subject"].iloc[0] == "P001"


def test_align_ratings_match():
    dfp = pd.DataFrame({
        "subject": ["P001"] * 3, "trial": [1, 2, 3],
        "code": [981, 585, 712], "valence": [5, 6, 4],
        "intensity": [6, 5, 1], "favourite": [5, 6, 4]})
    out = align_ratings([981, 585, 712], dfp)
    assert list(out["valence"]) == [5, 6, 4]
    assert list(out["intensity"]) == [6, 5, 1]


def test_align_ratings_mismatch_raises():
    dfp = pd.DataFrame({
        "subject": ["P001"] * 2, "trial": [1, 2], "code": [981, 585],
        "valence": [5, 6], "intensity": [6, 5], "favourite": [5, 6]})
    with pytest.raises(ValueError):
        align_ratings([981, 712], dfp)


def test_load_trial_order_real_file():
    from src.sensory import load_trial_order
    order = load_trial_order(ROOT / "protocol" / "experiment_sequences.xlsx")
    assert "P019" in order and "P001" in order
    assert len(order["P019"]) == 45
    # canonical P019 order (matches the sensory sheet, NOT the EEG markers)
    assert order["P019"][0] == 712
    assert order["P019"][1] == 902
    assert order["P019"][2] == 692
    # all codes are ints
    assert all(isinstance(c, int) for c in order["P019"])
