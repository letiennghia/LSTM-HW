import pytest
from imdb_sentiment.evaluation import compute_metrics
from imdb_sentiment.baselines import candidate_key


def test_known_metrics():
    metrics = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == .75
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(.8)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + .8) / 2)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert compute_metrics([0, 1], [0, 0])["precision"] == 0


def test_baseline_selection_tiebreaks():
    rows = [{"f1": .8, "accuracy": .8, "C": 1}, {"f1": .8, "accuracy": .9, "C": 10},
            {"f1": .8, "accuracy": .9, "C": .1}, {"f1": .7, "accuracy": 1., "C": .01}]
    assert max(rows, key=candidate_key)["C"] == .1
