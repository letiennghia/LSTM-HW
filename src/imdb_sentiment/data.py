"""Official splits with source IDs and a single text normalizer."""

from dataclasses import dataclass
import html
import re
import numpy as np
import pandas as pd
from datasets import load_dataset

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
SMOKE_SIZES = {"train": 256, "val": 64, "test": 128}


def clean_text(text: str, lowercase: bool = True) -> str:
    text = re.sub(r"<[^>]*>", " ", html.unescape(text))
    return " ".join((text.lower() if lowercase else text).split())


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


@dataclass
class DataBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    fingerprints: dict

    def splits(self):
        return ((name, getattr(self, name)) for name in ("train", "val", "test"))

    def manifest(self) -> pd.DataFrame:
        return pd.concat([frame[["review_id", "label"]].assign(split=name) for name, frame in self.splits()], ignore_index=True)


def load_data(config: dict, mode: str = "full", dataset=None) -> DataBundle:
    """The dataset argument permits offline tests with Hugging Face toy datasets."""
    if mode not in ("full", "smoke"):
        raise ValueError("mode must be full or smoke")
    if dataset is None:
        dataset = load_dataset(config["data"]["dataset_name"])
    official_train = dataset["train"].add_column("review_id", [f"train:{i}" for i in range(len(dataset["train"]))])
    official_test = dataset["test"].add_column("review_id", [f"test:{i}" for i in range(len(dataset["test"]))])
    split = official_train.train_test_split(test_size=0.2, seed=config["seed"], stratify_by_column="label")
    sources = {"train": split["train"], "val": split["test"], "test": official_test}
    frames = {}
    for name, source in sources.items():
        if mode == "smoke" and len(source) > SMOKE_SIZES[name]:
            source = source.train_test_split(train_size=SMOKE_SIZES[name], seed=config["seed"], stratify_by_column="label")["train"]
        frame = source.to_pandas()[["review_id", "text", "label"]]
        frame["text"] = frame["text"].map(lambda text: clean_text(text, config["data"]["lowercase"]))
        frame["token_count"] = frame["text"].map(lambda text: len(tokenize(text)))
        frames[name] = frame.reset_index(drop=True)
    if mode == "full":
        assert tuple(len(frames[n]) for n in sources) == (20000, 5000, 25000), "Full run requires 20k/5k/25k reviews"
    bundle = DataBundle(**frames, fingerprints={n: dataset[n]._fingerprint for n in ("train", "test")})
    manifest = bundle.manifest()
    assert manifest.review_id.is_unique, "Source IDs overlap across splits"
    assert set(manifest.label) == {0, 1}, "Only binary labeled reviews are permitted"
    return bundle


def describe_data(bundle: DataBundle, max_seq_len: int) -> pd.DataFrame:
    rows = []
    for name, frame in bundle.splits():
        lengths = frame.token_count
        rows.append({"split": name, "samples": len(frame), "negative": int((frame.label == 0).sum()),
                     "positive": int((frame.label == 1).sum()), "mean_tokens": float(lengths.mean()),
                     "median_tokens": float(lengths.median()), "p95_tokens": float(np.percentile(lengths, 95)),
                     "truncated_fraction": float((lengths > max_seq_len).mean())})
    return pd.DataFrame(rows)
