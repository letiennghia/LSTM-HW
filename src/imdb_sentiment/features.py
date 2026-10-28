"""Train-only sparse features and word sequences."""

from collections import Counter
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from .data import DataBundle, tokenize

PAD_ID, UNK_ID = 0, 1


def build_vocab(texts, max_vocab_size: int) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    words = sorted(counts, key=lambda word: (-counts[word], word))[:max_vocab_size - 2]
    return {"<PAD>": PAD_ID, "<UNK>": UNK_ID, **{word: i + 2 for i, word in enumerate(words)}}


def encode(text: str, vocab: dict, max_seq_len: int) -> list[int]:
    return [vocab.get(word, UNK_ID) for word in tokenize(text)[:max_seq_len]] or [UNK_ID]


class ReviewDataset(Dataset):
    def __init__(self, frame, vocab: dict, max_seq_len: int):
        self.sequences = [torch.tensor(encode(text, vocab, max_seq_len), dtype=torch.long) for text in frame.text]
        self.labels = torch.tensor(frame.label.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True, padding_value=PAD_ID), lengths, torch.stack(labels)


def make_loaders(bundle: DataBundle, vocab: dict, config: dict) -> dict:
    """Call before each neural model to reset the identical shuffle sequence."""
    generator = torch.Generator().manual_seed(config["seed"])
    data = config["data"]
    return {name: DataLoader(ReviewDataset(frame, vocab, data["max_seq_len"]), batch_size=data["batch_size"],
                            shuffle=name == "train", generator=generator if name == "train" else None,
                            num_workers=data.get("num_workers", 0), collate_fn=collate_batch)
            for name, frame in bundle.splits()}


@dataclass
class TfidfFeatures:
    vectorizer: TfidfVectorizer
    train: object
    val: object


def prepare_tfidf(bundle: DataBundle, config: dict, vectorizer=None) -> TfidfFeatures:
    if vectorizer is None:
        base = config["baselines"]
        vectorizer = TfidfVectorizer(max_features=base["max_features"], ngram_range=tuple(base["ngram_range"]),
                                     min_df=base["min_df"], sublinear_tf=base["sublinear_tf"], lowercase=False)
        train = vectorizer.fit_transform(bundle.train.text)
    else:
        train = vectorizer.transform(bundle.train.text)
    return TfidfFeatures(vectorizer, train, vectorizer.transform(bundle.val.text))
