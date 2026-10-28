"""Registered validation-only LSTM study and separately gated final evaluation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import pandas as pd
import torch
import yaml
from .evaluation import save_json, export_test
from .experiment import validate_run_id
from .features import build_vocab
from .models import create_model
from .optimization_config import load_study_config, candidate_config
from .optimization_data import load_training_partitions, load_final_test, make_loader
from .optimization_trainer import train_optimized
from .optimized_lstm import OptimizedLSTM
from .reproducibility import seed_everything, get_device, environment
from .trainer import train_neural, evaluate


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selection_key(row: dict) -> tuple:
    return -row["val_accuracy"], -row["val_f1"], row["val_loss"], row["experiment_id"]


def candidate_rows(directory: Path) -> list[dict]:
    metadata = read_json(directory / "study_metadata.json")
    rows = []
    for name in metadata["experiment_ids"]:
        folder = directory / name
        if not (folder / "metadata.json").exists():
            rows.append({"experiment_id": name, "status": "Not run"})
            continue
        record = read_json(folder / "metadata.json")
        if record.get("status") != "complete":
            rows.append({"experiment_id": name, "status": record.get("status", "incomplete")})
            continue
        metrics = read_json(folder / "validation_metrics.json")
        rows.append({"experiment_id": name, "status": "complete", "description": record["description"],
                     "val_accuracy": metrics["accuracy"], "val_f1": metrics["f1"], "val_loss": metrics["loss"],
                     "training_seconds": record["training_seconds"], "best_epoch": record["best_epoch"],
                     "epochs_trained": record["epochs_trained"], "parameter_count": record["parameter_count"],
                     "config_hash": record["config_hash"], "checkpoint_sha256": record["checkpoint_sha256"]})
    return rows


def freeze_selection(directory: Path) -> dict:
    """Write once, only after every registered candidate has completed."""
    if (directory / "selection.json").exists():
        raise FileExistsError("Selection is already frozen")
    metadata = read_json(directory / "study_metadata.json")
    rows = candidate_rows(directory)
    if len(rows) != 16 or any(row["status"] != "complete" for row in rows):
        raise ValueError("Cannot select: all 16 registered experiments must complete")
    for row in rows:
        if any(not torch.isfinite(torch.tensor(row[key])) for key in ("val_accuracy", "val_f1", "val_loss")):
            raise ValueError("Non-finite validation results")
        if file_hash(directory / row["experiment_id"] / "best_model.pt") != row["checkpoint_sha256"]:
            raise ValueError("Candidate checkpoint hash mismatch")
    winner = min(rows, key=selection_key)
    folder = directory / winner["experiment_id"]
    candidate_metadata = read_json(folder / "metadata.json")
    selection = {"experiment_id": winner["experiment_id"], "validation": winner,
                 "config": yaml.safe_load((folder / "config_used.yaml").read_text(encoding="utf-8")),
                 "checkpoint_sha256": winner["checkpoint_sha256"], "config_hash": winner["config_hash"],
                 "vocab_file": candidate_metadata["vocab_file"], "vocab_sha256": candidate_metadata["vocab_sha256"],
                 "control": candidate_metadata["control"], "split_hash": metadata["split_hash"],
                 "study_config_hash": metadata["study_config_hash"], "mode": metadata["mode"],
                 "ranking_rule": "validation accuracy descending, F1 descending, loss ascending, experiment ID ascending",
                 "frozen_at": datetime.now(timezone.utc).isoformat()}
    save_json(directory / "selection.json", selection)
    metadata.update(status="selected", selection_sha256=file_hash(directory / "selection.json"))
    save_json(directory / "study_metadata.json", metadata)
    return selection


def run_study(config: dict, mode: str = "full", study_id: str | None = None) -> Path:
    """Train and select without requesting the official test split."""
    if mode not in ("full", "smoke"):
        raise ValueError("mode must be full or smoke")
    reference = Path(config["reference_run"])
    control_config = yaml.safe_load((reference / "config_used.yaml").read_text(encoding="utf-8"))
    name = validate_run_id(study_id or f"{mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}")
    directory = Path(config["output_dir"]) / name
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite study: {directory}")
    train, val = load_training_partitions(config, mode, reference)
    directory.mkdir(parents=True)
    snapshot = directory / "reference"
    snapshot.mkdir()
    for filename in ("run_metadata.json", "split_manifest.csv", "config_used.yaml"):
        shutil.copyfile(reference / filename, snapshot / filename)
    manifest = pd.concat([train[["review_id", "label"]].assign(split="train"), val[["review_id", "label"]].assign(split="val")], ignore_index=True)
    manifest.to_csv(directory / "split_manifest.csv", index=False)
    (directory / "study_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = get_device(config.get("device", "auto"))
    metadata = {"study_id": name, "mode": mode, "status": "running", "seed": config["seed"],
                "split_sizes": {"train": len(train), "val": len(val)}, "split_hash": file_hash(directory / "split_manifest.csv"),
                "reference_run": str(reference.resolve()), "study_config_hash": object_hash(config),
                "reference_fingerprints": read_json(snapshot / "run_metadata.json")["fingerprints"],
                "experiment_ids": [item["id"] for item in config["experiments"]], "environment": environment(device)}
    save_json(directory / "study_metadata.json", metadata)
    print(f"Study: {directory.resolve()} | mode={mode} | train={len(train)} val={len(val)} | {device}", flush=True)
    vocabularies = {}
    for item in config["experiments"]:
        folder = directory / item["id"]
        folder.mkdir()
        effective = candidate_config(config, item, mode, control_config)
        (folder / "config_used.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
        print(f"\n--- {item['id']}: {item['description']} ---", flush=True)
        try:
            size = effective["data"]["max_vocab_size"]
            if size not in vocabularies:
                vocabularies[size] = build_vocab(train.text, size)
                save_json(directory / f"vocabulary_{size}.json", vocabularies[size])
            vocab = vocabularies[size]
            seed_everything(effective["seed"])
            loaders = {"train": make_loader(train, vocab, effective, training=True), "val": make_loader(val, vocab, effective)}
            model = create_model("lstm", len(vocab), effective) if item.get("control") else OptimizedLSTM(len(vocab), effective)
            if item.get("control"):
                model, details = train_neural(model, loaders, effective, folder, device)
                history = pd.read_csv(folder / "history.csv")
                history["learning_rate"] = effective["neural"]["learning_rate"]
                history.to_csv(folder / "history.csv", index=False)
                details["epochs_trained"] = len(history)
            else:
                details = train_optimized(model, loaders, effective, folder, device)
            vocab_name = f"vocabulary_{size}.json"
            save_json(folder / "metadata.json", {**details, "status": "complete", "description": item["description"],
                      "control": bool(item.get("control")), "config_hash": object_hash(effective),
                      "checkpoint_sha256": file_hash(folder / "best_model.pt"), "split_hash": metadata["split_hash"],
                      "vocab_file": vocab_name, "vocab_sha256": file_hash(directory / vocab_name),
                      "environment": metadata["environment"]})
            del model, loaders
            if device.type == "cuda":
                torch.cuda.empty_cache()
            pd.DataFrame(candidate_rows(directory)).to_csv(directory / "ablation_comparison.csv", index=False)
        except Exception as exc:
            save_json(folder / "metadata.json", {"status": "failed", "error": str(exc)})
            metadata.update(status="failed", failed_experiment=item["id"])
            save_json(directory / "study_metadata.json", metadata)
            pd.DataFrame(candidate_rows(directory)).to_csv(directory / "ablation_comparison.csv", index=False)
            raise
    winner = freeze_selection(directory)
    print(f"Frozen validation winner: {winner['experiment_id']}; test data has not been loaded.", flush=True)
    report_study(directory)
    return directory


def validate_selection(directory: Path) -> tuple[dict, dict]:
    metadata = read_json(directory / "study_metadata.json")
    if metadata["status"] not in ("selected", "evaluated") or not (directory / "selection.json").exists():
        raise ValueError("Final evaluation requires a frozen selection")
    selection = read_json(directory / "selection.json")
    if file_hash(directory / "selection.json") != metadata["selection_sha256"]:
        raise ValueError("Frozen selection hash mismatch")
    validate_run_id(selection["experiment_id"])
    if selection["experiment_id"] not in metadata["experiment_ids"]:
        raise ValueError("Selected experiment is not registered")
    if selection["split_hash"] != metadata["split_hash"] or file_hash(directory / "split_manifest.csv") != metadata["split_hash"]:
        raise ValueError("Training/validation manifest hash mismatch")
    if selection["study_config_hash"] != metadata["study_config_hash"] or selection["mode"] != metadata["mode"]:
        raise ValueError("Selected study identity mismatch")
    if object_hash(selection["config"]) != selection["config_hash"]:
        raise ValueError("Selected configuration hash mismatch")
    if file_hash(directory / selection["experiment_id"] / "best_model.pt") != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash mismatch")
    if Path(selection["vocab_file"]).name != selection["vocab_file"] or file_hash(directory / selection["vocab_file"]) != selection["vocab_sha256"]:
        raise ValueError("Selected vocabulary hash mismatch")
    return metadata, selection


def evaluate_selected(directory: str | Path) -> dict:
    directory = Path(directory)
    metadata, selection = validate_selection(directory)
    output = directory / "final"
    if output.exists():
        raise FileExistsError("Final evaluation already exists; refusing to overwrite")
    config = selection["config"]
    device = get_device(config.get("device", "auto"))
    vocab = read_json(directory / selection["vocab_file"])
    checkpoint = torch.load(directory / selection["experiment_id"] / "best_model.pt", map_location="cpu", weights_only=True)
    if checkpoint["config"] != config or checkpoint["vocab_size"] != len(vocab):
        raise ValueError("Checkpoint configuration or vocabulary size mismatch")
    model = create_model("lstm", len(vocab), config) if selection["control"] else OptimizedLSTM(len(vocab), config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    # Test data is requested only after frozen selection and artifact verification.
    frame = load_final_test(config, metadata["mode"], directory / "reference")
    loss, _, logits = evaluate(model, make_loader(frame, vocab, config), device)
    output.mkdir()
    metrics = export_test(output, frame, logits >= 0, torch.sigmoid(torch.tensor(logits)).numpy(), "probability")
    metrics["loss"] = loss
    save_json(output / "test_metrics.json", metrics)
    save_json(output / "metadata.json", {"experiment_id": selection["experiment_id"], "test_samples": len(frame),
              "selection_sha256": metadata["selection_sha256"], "checkpoint_sha256": selection["checkpoint_sha256"],
              "evaluated_at": datetime.now(timezone.utc).isoformat(), "environment": environment(device), "status": "complete"})
    if metadata["mode"] == "full":
        historical = []
        for name in ("logreg", "svm", "rnn", "lstm"):
            reference_folder = Path(metadata["reference_run"]) / name
            metrics_file = reference_folder / "test_metrics.json"
            if not metrics_file.exists():
                historical.append({"model": name, "status": "Not run"})
                continue
            old_metadata = read_json(reference_folder / "metadata.json")
            reference_metadata = read_json(directory / "reference" / "run_metadata.json")
            if old_metadata["manifest_hash"] != reference_metadata["manifest_hash"] or old_metadata["fingerprints"] != metadata["reference_fingerprints"]:
                raise ValueError("Historical benchmark split identity mismatch")
            historical.append({"model": name, "status": "historical benchmark", **read_json(metrics_file),
                               "training_seconds": old_metadata["training_seconds"], "parameter_count": old_metadata["parameter_count"]})
        save_json(directory / "historical_benchmarks.json", historical)
    metadata["status"] = "evaluated"
    save_json(directory / "study_metadata.json", metadata)
    report_study(directory)
    print(f"Selected model test accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}; {output.resolve()}", flush=True)
    return metrics


def report_study(directory: str | Path) -> pd.DataFrame:
    """Read saved artifacts only; no data loaders or training calls."""
    from .reporting import markdown_table
    directory = Path(directory)
    metadata = read_json(directory / "study_metadata.json")
    rows = candidate_rows(directory)
    table = pd.DataFrame(rows)
    anchor_id = metadata["experiment_ids"][1]
    complete = table.loc[table.status == "complete"].copy()
    if anchor_id in set(complete.experiment_id):
        anchor = complete.loc[complete.experiment_id == anchor_id].iloc[0]
        table["delta_accuracy_pp"] = (table.get("val_accuracy") - anchor.val_accuracy) * 100
        table["delta_f1"] = table.get("val_f1") - anchor.val_f1
        table["delta_loss"] = table.get("val_loss") - anchor.val_loss
    table.to_csv(directory / "ablation_comparison.csv", index=False)
    columns = [key for key in ["experiment_id", "status", "val_accuracy", "val_f1", "val_loss", "training_seconds", "best_epoch", "delta_accuracy_pp"] if key in table]
    mode_note = "SMOKE: execution checks only; no scientific conclusions." if metadata["mode"] == "smoke" else "Full 20k/5k validation study, seed 42."
    report = f"# LSTM optimization study\n\n{mode_note}\n\n" + markdown_table(table[columns]) + "\n\n"
    if (directory / "selection.json").exists():
        selection = read_json(directory / "selection.json")
        report += f"Frozen winner: **{selection['experiment_id']}**. Ranking: {selection['ranking_rule']}. Checkpoints are restored by validation loss; no test metric selected configurations.\n\n"
    report += "## Observed ablations\n\n"
    if "delta_accuracy_pp" in table:
        for row in table.to_dict(orient="records"):
            if row["status"] != "complete" or row["experiment_id"] in metadata["experiment_ids"][:2]:
                continue
            delta = row["delta_accuracy_pp"]
            outcome = "improved" if delta > 0 else "reduced" if delta < 0 else "tied"
            report += f"- {row['description']}: {outcome} validation accuracy relative to the anchor ({delta:+.2f} percentage points); F1 change {row['delta_f1']:+.4f}, loss change {row['delta_loss']:+.4f}; training {row['training_seconds']:.2f}s.\n"
    else:
        report += "The anchor is incomplete; no improvement claims are available.\n"
    report += "\nThe original control versus anchor changes a bundle of settings. The short-budget experiment changes both epochs and patience. Other ablations change one listed setting. Single-seed differences are descriptive, not significance tests; feature interactions are not measured.\n\n"
    report += f"Total completed candidate training time: {complete.training_seconds.sum() if 'training_seconds' in complete else 0:.2f}s, excluding preprocessing, selection and final test.\n\n"
    report += "## Final test\n\n"
    if (directory / "final" / "test_metrics.json").exists():
        final = read_json(directory / "final" / "test_metrics.json")
        selection = read_json(directory / "selection.json")
        final_row = {"model": f"selected:{selection['experiment_id']}", "status": metadata["mode"], **final,
                     "training_seconds": selection["validation"]["training_seconds"],
                     "parameter_count": selection["validation"]["parameter_count"]}
        historical_path = directory / "historical_benchmarks.json"
        final_table = pd.DataFrame((read_json(historical_path) if historical_path.exists() else []) + [final_row])
        final_table.drop(columns=["confusion_matrix"], errors="ignore").to_csv(directory / "final_comparison.csv", index=False)
        report += markdown_table(final_table[[key for key in ["model", "status", "accuracy", "f1", "macro_f1", "training_seconds"] if key in final_table]])
        report += "\n\nHistorical results are unchanged, on the same split. They were read for reporting only after selection and final evaluation. Timings compare different architectures/devices and exclude preprocessing.\n\n"
    else:
        report += "Not run. Test data is reserved until the configuration selection is frozen.\n\n"
    report += ("## Research basis and departures\n\n"
               "[Hameed & Garcia-Zapirain (2020)](https://doi.org/10.1109/ACCESS.2020.2988550) motivates BiLSTM mean/max pooling, regularization and evaluating 400 tokens; its GloVe embeddings and cross-validation differ from this study. "
               "[Sachan et al. (arXiv:2009.04007)](https://arxiv.org/abs/2009.04007) motivates max pooling, embedding/word dropout, vocabulary ablations and learning-rate decay; its pretrained embeddings and semi-supervised objectives are excluded here. "
               "The [Keras example](https://keras.io/examples/nlp/bidirectional_lstm_imdb/) illustrates bidirectional recurrence but uses a different validation protocol. "
               "Here embeddings are randomly initialized and trained only on 20k labeled training reviews; the official test split is never used for tuning. "
               "Head-tail truncation, exact dropout values, decay gamma and other starting hyperparameters are local hypotheses, not established optima. Published results are not directly reproducible under this repository's split or protocol.\n")
    (directory / "report.md").write_text(report, encoding="utf-8")
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "smoke"):
        child = commands.add_parser(name)
        child.add_argument("--config", default="configs/lstm_ablation.yaml")
        child.add_argument("--study-id")
    for name in ("evaluate", "report"):
        commands.add_parser(name).add_argument("--study", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ("run", "smoke"):
            directory = run_study(load_study_config(args.config), "smoke" if args.command == "smoke" else "full", args.study_id)
            if args.command == "smoke":
                evaluate_selected(directory)
        elif args.command == "evaluate":
            evaluate_selected(args.study)
        else:
            print(report_study(args.study).to_string(index=False))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
