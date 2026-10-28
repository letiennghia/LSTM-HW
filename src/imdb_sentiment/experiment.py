"""Shared orchestration used by the notebook and CLI."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid
import joblib
import torch
import yaml
from .config import validate_config
from .data import DataBundle, describe_data
from .evaluation import export_test, save_json
from .features import build_vocab, prepare_tfidf, make_loaders
from .reproducibility import seed_everything, get_device, environment

MODEL_NAMES = ("logreg", "svm", "rnn", "lstm")


def config_signature(config: dict, mode: str) -> str:
    settings = {k: v for k, v in config.items() if k != "output_dir"}
    return hashlib.sha256(json.dumps({"config": settings, "mode": mode}, sort_keys=True).encode()).hexdigest()


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id):
        raise ValueError("run-id must be a simple name containing letters, digits, _, - or .")
    return run_id


class Experiment:
    """A single immutable split/config context with explicit per-model execution."""

    def __init__(self, config: dict, bundle: DataBundle, mode: str = "full", run_id: str | None = None):
        from .visualization import plot_lengths
        validate_config(config)
        self.config, self.bundle, self.mode = config, bundle, mode
        self.device = get_device(config.get("device", "auto"))
        generated_id = f"{mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        self.run_id = validate_run_id(run_id or generated_id)
        self.directory = Path(config["output_dir"]) / self.run_id
        manifest = bundle.manifest()
        signature = config_signature(config, mode)
        manifest_hash = hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest()
        self.identity = {"config_signature": signature, "manifest_hash": manifest_hash, "fingerprints": bundle.fingerprints}
        self.metadata = {"run_id": self.run_id, "mode": mode, "seed": config["seed"], "dataset": config["data"]["dataset_name"],
                         "split_sizes": {name: len(frame) for name, frame in bundle.splits()}, **self.identity,
                         "created_at": datetime.now(timezone.utc).isoformat(), "environment": environment(self.device)}
        if self.directory.exists():
            saved = json.loads((self.directory / "run_metadata.json").read_text(encoding="utf-8"))
            if any(saved.get(k) != v for k, v in self.identity.items()):
                raise ValueError("Existing run has incompatible configuration, dataset or sample IDs")
            self.metadata = saved
        else:
            self.directory.mkdir(parents=True)
            save_json(self.directory / "run_metadata.json", self.metadata)
            (self.directory / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            manifest.to_csv(self.directory / "split_manifest.csv", index=False)
            bundle.test[["review_id", "text", "token_count"]].to_csv(self.directory / "test_reviews.csv", index=False)
            describe_data(bundle, config["data"]["max_seq_len"]).to_csv(self.directory / "data_statistics.csv", index=False)
            plot_lengths(bundle, self.directory / "review_lengths.png")
        self.tfidf = None
        self.vocab = None
        print(f"{mode.upper()} run: {self.directory.resolve()}\nDevice: {self.device}", flush=True)
        print(describe_data(bundle, config["data"]["max_seq_len"]).to_string(index=False), flush=True)

    def prepare_baselines(self):
        if self.tfidf is None:
            path = self.directory / "tfidf_vectorizer.joblib"
            # Artifacts are loaded only from this user-controlled experiment directory.
            vectorizer = joblib.load(path) if path.exists() else None
            self.tfidf = prepare_tfidf(self.bundle, self.config, vectorizer)
            if not path.exists():
                joblib.dump(self.tfidf.vectorizer, path)
        return self.tfidf

    def prepare_neural(self):
        if self.vocab is None:
            path = self.directory / "vocabulary.json"
            if path.exists():
                self.vocab = json.loads(path.read_text(encoding="utf-8"))
            else:
                self.vocab = build_vocab(self.bundle.train.text, self.config["data"]["max_vocab_size"])
                save_json(path, self.vocab)
        return self.vocab

    def train(self, name: str):
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {name}")
        directory = self.directory / name
        if directory.exists():
            raise FileExistsError(f"Refusing to replace {directory}. Start a new run, or use a run ID with this model missing.")
        directory.mkdir()
        seed_everything(self.config["seed"])
        (directory / "config_used.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        if name in ("logreg", "svm"):
            from .baselines import train_baseline
            features = self.prepare_baselines()
            model, details = train_baseline(name, features, self.bundle, self.config, directory)
            test_features = features.vectorizer.transform(self.bundle.test.text)
            predicted = model.predict(test_features)
            scores = model.predict_proba(test_features)[:, 1] if name == "logreg" else model.decision_function(test_features)
            score_type = "probability" if name == "logreg" else "margin"
            model_device = torch.device("cpu")
        else:
            from .models import create_model
            from .trainer import train_neural, evaluate
            vocab = self.prepare_neural()
            loaders = make_loaders(self.bundle, vocab, self.config)
            model = create_model(name, len(vocab), self.config)
            print(f"{name}: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters", flush=True)
            model, details = train_neural(model, loaders, self.config, directory, self.device)
            _, _, logits = evaluate(model, loaders["test"], self.device)
            predicted = logits >= 0
            scores = torch.sigmoid(torch.as_tensor(logits)).numpy()
            score_type, model_device = "probability", self.device
        metrics = export_test(directory, self.bundle.test, predicted, scores, score_type)
        metadata = {**self.metadata, **details, "model": name, "status": "complete", "score_type": score_type,
                    "environment": environment(model_device)}
        save_json(directory / "metadata.json", metadata)
        print(f"{name}: test F1={metrics['f1']:.4f}; artifacts: {directory.resolve()}", flush=True)
        return model, metrics, metadata
