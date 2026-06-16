import numpy as np

from src.pseudo_subjects import (
    average_classwise_blocks,
    make_classwise_pseudo_subjects,
    pseudo_subject_report,
)


def test_make_classwise_pseudo_subjects_splits_each_label_into_blocks():
    y = np.array(["Arabica"] * 15 + ["Robusta"] * 15, dtype=object)
    subjects = np.array(["P001"] * 30, dtype=object)

    pseudo, real, labels, block_index = make_classwise_pseudo_subjects(
        y, subjects, block_size=5)

    assert np.array_equal(real, subjects)
    assert np.array_equal(labels, y)
    assert set(pseudo) == {
        "P001_Arabica_B01",
        "P001_Arabica_B02",
        "P001_Arabica_B03",
        "P001_Robusta_B01",
        "P001_Robusta_B02",
        "P001_Robusta_B03",
    }
    assert np.all(pseudo[:5] == "P001_Arabica_B01")
    assert np.all(pseudo[10:15] == "P001_Arabica_B03")
    assert np.all(pseudo[15:20] == "P001_Robusta_B01")
    assert np.array_equal(block_index[:15], np.repeat([1, 2, 3], 5))
    assert np.array_equal(block_index[15:], np.repeat([1, 2, 3], 5))


def test_make_classwise_pseudo_subjects_keeps_partial_last_block():
    y = np.array(["Arabica"] * 6 + ["Robusta"] * 4, dtype=object)
    subjects = np.array(["P001"] * 10, dtype=object)

    pseudo, _, _, block_index = make_classwise_pseudo_subjects(
        y, subjects, block_size=5)

    assert np.sum(pseudo == "P001_Arabica_B01") == 5
    assert np.sum(pseudo == "P001_Arabica_B02") == 1
    assert np.sum(pseudo == "P001_Robusta_B01") == 4
    assert block_index[5] == 2


def test_pseudo_subject_report_counts_labels_and_epochs():
    y = np.array(["Arabica"] * 5 + ["Robusta"] * 5, dtype=object)
    subjects = np.array(["P001"] * 10, dtype=object)
    pseudo, real, labels, block_index = make_classwise_pseudo_subjects(
        y, subjects, block_size=5)

    report = pseudo_subject_report(real, pseudo, labels, block_index)

    assert list(report["pseudo_subject"]) == [
        "P001_Arabica_B01",
        "P001_Robusta_B01",
    ]
    assert list(report["n_epochs"]) == [5, 5]
    assert list(report["label"]) == ["Arabica", "Robusta"]


def test_average_classwise_blocks_averages_complete_blocks_only():
    X = np.arange(12 * 2 * 3, dtype=float).reshape(12, 2, 3)
    y = np.array(["Arabica"] * 6 + ["Robusta"] * 6, dtype=object)
    subjects = np.array(["P001"] * 12, dtype=object)

    avg = average_classwise_blocks(X, y, subjects, block_size=5,
                                   drop_incomplete=True)

    assert avg["X"].shape == (2, 2, 3)
    assert avg["y"].tolist() == ["Arabica", "Robusta"]
    assert avg["real_subjects"].tolist() == ["P001", "P001"]
    assert avg["pseudo_subjects"].tolist() == [
        "P001_Arabica_B01",
        "P001_Robusta_B01",
    ]
    assert avg["n_epochs"].tolist() == [5, 5]
    assert np.allclose(avg["X"][0], X[:5].mean(axis=0))
    assert np.allclose(avg["X"][1], X[6:11].mean(axis=0))


def test_average_classwise_blocks_can_keep_incomplete_blocks():
    X = np.arange(6 * 1 * 2, dtype=float).reshape(6, 1, 2)
    y = np.array(["Arabica"] * 6, dtype=object)
    subjects = np.array(["P001"] * 6, dtype=object)

    avg = average_classwise_blocks(X, y, subjects, block_size=5,
                                   drop_incomplete=False)

    assert avg["X"].shape == (2, 1, 2)
    assert avg["pseudo_subjects"].tolist() == [
        "P001_Arabica_B01",
        "P001_Arabica_B02",
    ]
    assert avg["n_epochs"].tolist() == [5, 1]
