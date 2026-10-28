"""Load training partitions without fetching, encoding or evaluating test reviews."""

from pathlib import Path
import json
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from .data import clean_text, tokenize
from .features import ReviewDataset, collate_batch, UNK_ID


def truncate_tokens(tokens: list, limit: int, strategy: str) -> list:
    if limit < 1 or strategy not in ("head", "head_tail"):
        raise ValueError("Use a positive limit and head or head_tail truncation")
    if len(tokens) <= limit:
        return tokens
    if strategy == "head":
        return tokens[:limit]
    tail = limit // 2
    return tokens[:limit - tail] + (tokens[-tail:] if tail else [])


class OptimizedReviewDataset(ReviewDataset):
    def __init__(self, frame: pd.DataFrame, vocab: dict, config: dict):
        data = config["data"]
        self.sequences = []
        for text in frame.text:
            tokens = truncate_tokens(tokenize(text), data["max_seq_len"], data.get("truncation", "head"))
            self.sequences.append(torch.tensor([vocab.get(word, UNK_ID) for word in tokens] or [UNK_ID], dtype=torch.long))
        self.labels = torch.tensor(frame.label.to_numpy(), dtype=torch.float32)


def frame_from_source(source, lowercase: bool) -> pd.DataFrame:
    frame = source.to_pandas()[["review_id", "text", "label"]]
    frame["text"] = frame.text.map(lambda text: clean_text(text, lowercase))
    return frame.reset_index(drop=True)


def verify_partition(source, name: str, reference: Path) -> None:
    manifest = pd.read_csv(reference / "split_manifest.csv")
    expected = manifest.loc[manifest.split == name, ["review_id", "label"]].sort_values("review_id").reset_index(drop=True)
    actual = source.to_pandas()[["review_id", "label"]].sort_values("review_id").reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError(f"{name} IDs or labels differ from the preserved benchmark")


def verify_fingerprint(source, name: str, reference: Path) -> None:
    metadata = json.loads((reference / "run_metadata.json").read_text(encoding="utf-8"))
    if source._fingerprint != metadata["fingerprints"][name]:
        raise ValueError(f"Official {name} dataset fingerprint differs from the benchmark")


def load_training_partitions(config: dict, mode: str, reference: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = load_dataset(config["data"]["dataset_name"], split="train")
    verify_fingerprint(source, "train", reference)
    source = source.add_column("review_id", [f"train:{i}" for i in range(len(source))])
    parts = source.train_test_split(test_size=0.2, seed=config["seed"], stratify_by_column="label")
    result = []
    for name, source in [("train", parts["train"]), ("val", parts["test"])]:
        verify_partition(source, name, reference)
        if mode == "full" and len(source) != {"train": 20000, "val": 5000}[name]:
            raise ValueError("Full ablation requires 20k/5k training/validation reviews")
        limit = {"train": 256, "val": 64}[name]
        if mode == "smoke" and len(source) > limit:
            source = source.train_test_split(train_size=limit, seed=config["seed"], stratify_by_column="label")["train"]
        result.append(frame_from_source(source, config["data"]["lowercase"]))
    return tuple(result)


def load_final_test(config: dict, mode: str, reference: Path) -> pd.DataFrame:
    source = load_dataset(config["data"]["dataset_name"], split="test")
    verify_fingerprint(source, "test", reference)
    source = source.add_column("review_id", [f"test:{i}" for i in range(len(source))])
    verify_partition(source, "test", reference)
    if mode == "full" and len(source) != 25000:
        raise ValueError("Final test requires 25k reviews")
    if mode == "smoke" and len(source) > 128:
        source = source.train_test_split(train_size=128, seed=config["seed"], stratify_by_column="label")["train"]
    return frame_from_source(source, config["data"]["lowercase"])


def make_loader(frame: pd.DataFrame, vocab: dict, config: dict, training: bool = False) -> DataLoader:
    return DataLoader(OptimizedReviewDataset(frame, vocab, config), batch_size=config["data"]["batch_size"],
                      shuffle=training, generator=torch.Generator().manual_seed(config["seed"]) if training else None,
                      num_workers=config["data"].get("num_workers", 0), collate_fn=collate_batch)
