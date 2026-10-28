"""Compare saved results without training or accessing the dataset service."""

import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from .evaluation import save_json, compute_metrics

MODEL_NAMES = ("logreg", "svm", "rnn", "lstm")
METRICS = ("accuracy", "precision", "recall", "f1", "macro_f1")


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if pd.isna(value):
            return "Not run"
        return f"{value:.4f}" if isinstance(value, float) else str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def find_run(config: dict, run_id: str | None = None) -> Path:
    from .experiment import config_signature, validate_run_id
    root = Path(config["output_dir"])
    if run_id:
        directory = root / validate_run_id(run_id)
        meta_path = directory / "run_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        from .config import effective_config
        expected = config_signature(effective_config(config, meta["mode"]), meta["mode"])
        if meta["config_signature"] != expected:
            raise ValueError("Requested run is incompatible with the supplied configuration")
        return directory
    matches = []
    for meta_path in root.glob("*/run_metadata.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["mode"] == "full" and meta["config_signature"] == config_signature(config, "full"):
            matches.append((meta["created_at"], meta_path.parent))
    if not matches:
        raise FileNotFoundError("No compatible full run exists. Run training first or specify --run-id for a smoke run.")
    return max(matches, key=lambda item: item[0])[1]


def compare_results(directory: str | Path) -> pd.DataFrame:
    from .visualization import plot_comparison
    directory = Path(directory)
    run = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(directory / "split_manifest.csv")
    expected = manifest.loc[manifest.split == "test", ["review_id", "label"]].set_index("review_id").sort_index()
    rows, missing, predictions, histories = [], [], {}, {}
    for name in MODEL_NAMES:
        model_dir = directory / name
        if not (model_dir / "metadata.json").exists():
            missing.append(name)
            continue
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        for key in ("config_signature", "manifest_hash", "fingerprints", "run_id", "mode", "split_sizes"):
            if metadata.get(key) != run.get(key):
                raise ValueError(f"Incompatible {name} artifact: {key}")
        if metadata.get("status") != "complete":
            missing.append(name)
            continue
        metrics = json.loads((model_dir / "test_metrics.json").read_text(encoding="utf-8"))
        if any(not isinstance(metrics.get(k), (float, int)) or not np.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1 for k in METRICS):
            raise ValueError(f"Invalid metric artifact for {name}")
        prediction = pd.read_csv(model_dir / "test_predictions.csv")
        if not prediction.review_id.is_unique or set(prediction.review_id) != set(expected.index):
            raise ValueError(f"Prediction IDs do not match the test manifest: {name}")
        prediction = prediction.set_index("review_id").sort_index()
        if not np.array_equal(prediction.true_label.to_numpy(), expected.label.to_numpy()):
            raise ValueError(f"Prediction labels do not match the test manifest: {name}")
        if not set(prediction.predicted_label).issubset({0, 1, False, True}):
            raise ValueError(f"Invalid predicted labels: {name}")
        recomputed = compute_metrics(prediction.true_label, prediction.predicted_label.astype(int))
        if any(not np.isclose(metrics[k], recomputed[k], rtol=0, atol=1e-12) for k in METRICS):
            raise ValueError(f"Saved metrics disagree with predictions: {name}")
        predictions[name] = prediction.predicted_label.astype(int)
        rows.append({"model": name, "status": "complete", **{k: metrics[k] for k in METRICS},
                     "training_seconds": metadata["training_seconds"], "parameter_count": metadata["parameter_count"],
                     "device": metadata["environment"]["device"],
                     "best_config": json.dumps(metadata.get("best_hyperparameters", {"best_epoch": metadata.get("best_epoch")}))})
        if (model_dir / "history.csv").exists():
            histories[name] = pd.read_csv(model_dir / "history.csv")
    if not rows:
        raise FileNotFoundError(f"No completed model results. Missing models: {', '.join(missing)}")
    if missing:
        warnings.warn(f"Missing models: {', '.join(missing)}", stacklevel=2)
    completed = pd.DataFrame(rows)
    table = pd.DataFrame([{**{"model": name, "status": "Not run"}, **{k: None for k in METRICS}} for name in missing])
    table = pd.concat([completed, table], ignore_index=True) if missing else completed
    output = directory / "comparison"
    output.mkdir(exist_ok=True)
    table.to_csv(output / "comparison.csv", index=False)
    save_json(output / "status.json", {"mode": run["mode"], "missing_models": missing, "run_id": run["run_id"]})
    plot_comparison(completed, histories, output)
    errors = expected.rename(columns={"label": "true_label"}).copy()
    for name, series in predictions.items():
        errors[name] = series
    wrong = np.logical_or.reduce([(errors[name] != errors.true_label).to_numpy() for name in predictions])
    errors = errors.loc[wrong].head(10).reset_index()
    reviews = pd.read_csv(directory / "test_reviews.csv")
    reviews["excerpt"] = reviews.text.fillna("").str.slice(0, 500)
    errors = errors.merge(reviews[["review_id", "excerpt"]], on="review_id", how="left", validate="one_to_one")
    errors.to_csv(output / "error_analysis.csv", index=False)
    best = completed.sort_values(["f1", "accuracy"], ascending=False).iloc[0]
    caveat = "SMOKE RESULTS: pipeline verification only; these numbers do not support scientific conclusions." if run["mode"] == "smoke" else "Full IMDB evaluation on the fixed official test split."
    report = (f"# IMDB model comparison\n\nRun: `{run['run_id']}`. {caveat}\n\n"
              + markdown_table(table) + f"\n\nAmong completed models, **{best['model']}** has the highest measured test F1 ({best['f1']:.4f}). "
              + (f"Comparison is incomplete: missing {', '.join(missing)}. " if missing else "")
              + "This is an observed single-seed result, not a claim of statistical significance. Test results were not used to tune models.\n\n"
              + "Linear models use sparse word/bigram TF-IDF; neural models use learned word embeddings and truncated sequences. "
              + "RNN and LSTM share dimensions and training rules, but LSTM has more trainable parameters. "
              + "Baseline time includes the C search; neural time includes epochs, validation and checkpointing. Shared feature preparation and test evaluation are excluded. "
              + "CPU/GPU timing differences limit direct cost comparisons.\n\n"
              + "Vanishing gradients, truncation of review endings and overfitting can affect neural results. "
              + "Loss and validation F1 curves describe observed optimization; no architecture is assumed best in advance.\n\n"
              + "## Error examples\n\n" + (markdown_table(errors) if len(errors) else "No errors among the available predictions.") + "\n")
    (output / "comparison.md").write_text(report, encoding="utf-8")
    print(f"Comparison saved: {output.resolve()}", flush=True)
    return table
