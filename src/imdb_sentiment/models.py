"""Unidirectional RNN and LSTM binary classifiers."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentimentRNN(nn.Module):
    recurrent_class = nn.RNN

    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.recurrent = self.recurrent_class(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(self.embedding(input_ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.recurrent(packed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.classifier(self.dropout(hidden[-1])).squeeze(1)


class SentimentLSTM(SentimentRNN):
    recurrent_class = nn.LSTM


def create_model(name: str, vocab_size: int, config: dict) -> nn.Module:
    settings = config["neural"]
    cls = {"rnn": SentimentRNN, "lstm": SentimentLSTM}[name]
    return cls(vocab_size, **{key: settings[key] for key in ("embedding_dim", "hidden_dim", "num_layers", "dropout")})
