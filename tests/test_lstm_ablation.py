import pandas as pd
import pytest
import torch
import yaml
from imdb_sentiment.optimization_config import load_study_config, candidate_config, validate_optimized
from imdb_sentiment import lstm_ablation as study
from imdb_sentiment.optimization_trainer import train_optimized
from imdb_sentiment.optimization_data import make_loader
from imdb_sentiment.optimized_lstm import OptimizedLSTM


@pytest.fixture
def study_config(tmp_path, toy_dataset, config):
    reference = tmp_path / "reference"
    reference.mkdir()
    from imdb_sentiment.data import load_data
    bundle = load_data(config, "smoke", toy_dataset)
    bundle.manifest().to_csv(reference / "split_manifest.csv", index=False)
    (reference / "config_used.yaml").write_text(yaml.safe_dump(config))
    study.save_json(reference / "run_metadata.json", {"fingerprints": bundle.fingerprints, "manifest_hash": "toy"})
    result = load_study_config("configs/lstm_ablation.yaml")
    result.update(reference_run=str(reference), output_dir=str(tmp_path / "studies"), device="cpu")
    return result


def test_registration_and_independent_overrides():
    config = load_study_config("configs/lstm_ablation.yaml")
    assert len(config["experiments"]) == 16
    assert config["neural"]["pooling"] == "mean_max" and config["neural"]["bidirectional"]
    changed = candidate_config(config, config["experiments"][4])
    assert changed["data"]["max_seq_len"] == 500 and config["data"]["max_seq_len"] == 400
    for item in config["experiments"][2:]:
        effective = candidate_config(config, item)
        differences = [(section, key) for section in ("data", "neural") for key in config[section] if effective[section][key] != config[section][key]]
        assert len(differences) == (2 if item["id"] == "16_short_budget" else 1)


@pytest.mark.parametrize("key,value", [("pooling", "bad"), ("word_dropout", -.1), ("embedding_dropout", 1.0), ("scheduler_gamma", 0), ("bidirectional", "yes")])
def test_optimization_validation(key, value):
    config = load_study_config("configs/lstm_ablation.yaml")
    config["neural"][key] = value
    with pytest.raises(ValueError):
        validate_optimized(config)


def test_selection_key_uses_only_validation():
    rows = [{"experiment_id": "b", "val_accuracy": .8, "val_f1": .8, "val_loss": .3, "test_accuracy": 1},
            {"experiment_id": "a", "val_accuracy": .8, "val_f1": .9, "val_loss": .4, "test_accuracy": 0}]
    assert min(rows, key=study.selection_key)["experiment_id"] == "a"
    rows[0]["val_f1"] = .9
    assert min(rows, key=study.selection_key)["experiment_id"] == "b"
    rows[0]["val_loss"] = .4
    assert min(rows, key=study.selection_key)["experiment_id"] == "a"


def test_study_never_loads_test_until_selection(study_config, toy_dataset, monkeypatch):
    import imdb_sentiment.optimization_data as data
    calls = []
    def load(name, *, split):
        calls.append(split)
        if split != "train":
            raise AssertionError("Test data accessed during ablation")
        return toy_dataset["train"]
    monkeypatch.setattr(data, "load_dataset", load)
    output = study.run_study(study_config, "smoke", "toy")
    assert calls == ["train"]
    selection = study.read_json(output / "selection.json")
    rows = study.candidate_rows(output)
    assert len(rows) == 16 and all(row["status"] == "complete" for row in rows)
    assert selection["experiment_id"] == min(rows, key=study.selection_key)["experiment_id"]
    assert not (output / "final").exists()
    with pytest.raises(FileExistsError):
        study.run_study(study_config, "smoke", "toy")
    with pytest.raises(FileExistsError):
        study.freeze_selection(output)
    selection_path = output / "selection.json"
    original_selection = selection_path.read_bytes()
    selection_path.write_bytes(original_selection + b" ")
    with pytest.raises(ValueError, match="selection hash"):
        study.validate_selection(output)
    selection_path.write_bytes(original_selection)
    monkeypatch.setattr(data, "load_dataset", lambda name, *, split: toy_dataset[split])
    # Reporting cannot load any data, even if called before final evaluation.
    monkeypatch.setattr(study, "load_final_test", lambda *args: (_ for _ in ()).throw(AssertionError("Report accessed test")))
    study.report_study(output)
    monkeypatch.setattr(study, "load_final_test", data.load_final_test)
    metrics = study.evaluate_selected(output)
    assert 0 <= metrics["accuracy"] <= 1
    assert len(pd.read_csv(output / "final/test_predictions.csv")) == 20
    with pytest.raises(FileExistsError):
        study.evaluate_selected(output)
    checkpoint = output / selection["experiment_id"] / "best_model.pt"
    with checkpoint.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checkpoint hash"):
        study.validate_selection(output)


def test_missing_candidates_and_unfrozen_evaluation(tmp_path):
    study.save_json(tmp_path / "study_metadata.json", {"status": "running", "mode": "smoke", "experiment_ids": [f"x{i}" for i in range(16)]})
    with pytest.raises(ValueError, match="all 16"):
        study.freeze_selection(tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        study.evaluate_selected(tmp_path)
    table = study.report_study(tmp_path)
    assert table.status.eq("Not run").all()


def test_optimization_notebook_compiles():
    import ast
    import nbformat
    notebook = nbformat.read("imdb_lstm_optimization.ipynb", as_version=4)
    nbformat.validate(notebook)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert sum('MODE = "full"' in source for source in sources) == 1
    for source in sources:
        ast.parse(source)


def test_scheduler_and_restoration(study_config, bundle, tmp_path):
    from imdb_sentiment.features import build_vocab
    config = candidate_config(study_config, study_config["experiments"][1], "smoke")
    config["neural"].update(epochs=3, scheduler="exponential", early_stopping_min_delta=10, early_stopping_patience=2)
    vocab = build_vocab(bundle.train.text, 2000)
    model = OptimizedLSTM(len(vocab), config)
    loaders = {"train": make_loader(bundle.train, vocab, config, True), "val": make_loader(bundle.val, vocab, config)}
    details = train_optimized(model, loaders, config, tmp_path, torch.device("cpu"))
    history = pd.read_csv(tmp_path / "history.csv")
    assert history.learning_rate.tolist() == pytest.approx([.0005, .0005 * .95, .0005 * .95**2])
    assert details["best_epoch"] == 1 and details["checkpoint_restored"]
    saved = torch.load(tmp_path / "best_model.pt", weights_only=True)
    assert all(torch.equal(value, saved["state_dict"][key]) for key, value in model.state_dict().items())
