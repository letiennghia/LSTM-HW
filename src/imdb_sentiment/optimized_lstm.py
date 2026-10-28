"""Separate LSTM track; the original benchmark model classes remain untouched."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .features import PAD_ID, UNK_ID


def masked_pool(outputs: torch.Tensor, lengths: torch.Tensor, strategy: str) -> torch.Tensor:
    """Pool valid recurrent outputs, including features that are all negative."""
    mask = torch.arange(outputs.size(1), device=outputs.device)[None, :] < lengths.to(outputs.device)[:, None]
    maximum = outputs.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
    if strategy == "max":
        return maximum
    if strategy != "mean_max":
        raise ValueError(f"Unsupported pooling: {strategy}")
    mean = outputs.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / lengths.to(outputs.device).unsqueeze(1)
    return torch.cat([mean, maximum], dim=1)


def final_hidden(hidden: torch.Tensor, num_layers: int, bidirectional: bool) -> torch.Tensor:
    """Concatenate the final layer's directional states, preserving batch order."""
    directions = 2 if bidirectional else 1
    last = hidden.reshape(num_layers, directions, hidden.size(1), hidden.size(2))[-1]
    return last.transpose(0, 1).reshape(hidden.size(1), -1)


class OptimizedLSTM(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        net = config["neural"]
        self.num_layers, self.bidirectional = net["num_layers"], net["bidirectional"]
        self.pooling, self.word_dropout = net["pooling"], net["word_dropout"]
        self.embedding = nn.Embedding(vocab_size, net["embedding_dim"], padding_idx=PAD_ID)
        self.embedding_dropout = nn.Dropout(net["embedding_dropout"])
        self.recurrent = nn.LSTM(net["embedding_dim"], net["hidden_dim"], num_layers=self.num_layers,
                                 bidirectional=self.bidirectional, batch_first=True)
        width = net["hidden_dim"] * (2 if self.bidirectional else 1) * (2 if self.pooling == "mean_max" else 1)
        self.classifier_dropout = nn.Dropout(net["classifier_dropout"])
        self.classifier = nn.Linear(width, 1)

    def drop_words(self, ids: torch.Tensor) -> torch.Tensor:
        if not self.training or self.word_dropout == 0:
            return ids
        replace = (ids > UNK_ID) & (torch.rand(ids.shape, device=ids.device) < self.word_dropout)
        return ids.masked_fill(replace, UNK_ID)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_dropout(self.embedding(self.drop_words(ids)))
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, _) = self.recurrent(packed)
        if self.pooling == "final":
            representation = final_hidden(hidden, self.num_layers, self.bidirectional)
        else:
            outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=ids.size(1))
            representation = masked_pool(outputs, lengths, self.pooling)
        return self.classifier(self.classifier_dropout(representation)).squeeze(1)
