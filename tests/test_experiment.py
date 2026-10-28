import json
import pandas as pd
import pytest
from imdb_sentiment.experiment import Experiment
from imdb_sentiment.reporting import compare_results, find_run


def test_end_to_end_and_saved_alignment(config, bundle):
    experiment = Experiment(config, bundle, "smoke", "test-run")
    with pytest.raises(FileNotFoundError, match="Missing models"):
        compare_results(experiment.directory)
    for name in ("logreg", "svm", "rnn", "lstm"):
        experiment.train(name)
    # A reordered prediction CSV must still compare by review ID.
    path = experiment.directory / "svm" / "test_predictions.csv"
    pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
    table = compare_results(experiment.directory)
    assert len(table) == 4 and table.status.eq("complete").all()
    assert (experiment.directory / "comparison" / "comparison.md").exists()
    for name in ("rnn", "lstm"):
        assert (experiment.directory / name / "history.csv").exists()
    with pytest.raises(FileExistsError):
        experiment.train("logreg")
    assert find_run(config, "test-run") == experiment.directory
    with pytest.raises(FileNotFoundError, match="compatible full"):
        find_run(config)
    prediction = pd.read_csv(path)
    prediction.loc[0, "true_label"] = 1 - prediction.loc[0, "true_label"]
    prediction.to_csv(path, index=False)
    with pytest.raises(ValueError, match="labels do not match"):
        compare_results(experiment.directory)


def test_missing_models_and_incompatible_results(config, bundle):
    experiment = Experiment(config, bundle, "smoke", "partial")
    experiment.train("logreg")
    with pytest.warns(UserWarning, match="Missing models"):
        table = compare_results(experiment.directory)
    assert table.loc[table.status == "Not run", "f1"].isna().all()
    metadata_path = experiment.directory / "logreg" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["manifest_hash"] = "wrong"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Incompatible"):
        compare_results(experiment.directory)
    config["seed"] = 10
    with pytest.raises(ValueError, match="incompatible"):
        Experiment(config, bundle, "smoke", "partial")


def test_comparison_rejects_stale_metrics(config, bundle):
    experiment = Experiment(config, bundle, "smoke", "stale")
    experiment.train("logreg")
    path = experiment.directory / "logreg" / "test_metrics.json"
    metrics = json.loads(path.read_text())
    metrics["accuracy"] = 0.123
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="disagree"):
        compare_results(experiment.directory)


def test_cli_help_and_compare_never_load_dataset(monkeypatch, config, tmp_path):
    import imdb_sentiment.data
    from imdb_sentiment.cli import main
    def fail(*args, **kwargs):
        raise AssertionError("Dataset loading must not occur")
    monkeypatch.setattr(imdb_sentiment.data, "load_data", fail)
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    import yaml
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit) as compare_exit:
        main(["compare", "--config", str(path)])
    assert compare_exit.value.code == 1


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "..", "C:\\path"])
def test_run_id_is_not_a_path(config, bundle, run_id):
    with pytest.raises(ValueError, match="run-id"):
        Experiment(config, bundle, "smoke", run_id)
