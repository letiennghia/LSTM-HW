import torch
from imdb_sentiment.data import tokenize
from imdb_sentiment.features import build_vocab, encode, collate_batch, prepare_tfidf, make_loaders


def test_tokenize_vocabulary_and_encoding():
    assert tokenize("isn't bad; don't stop") == ["isn't", "bad", "don't", "stop"]
    vocab = build_vocab(["good good not bad", "good"], 5)
    assert len(vocab) == 5 and vocab["<PAD>"] == 0 and vocab["<UNK>"] == 1
    assert encode("unseen", vocab, 10) == [1]
    assert encode("", vocab, 10) == [1]
    assert len(encode("good good good", vocab, 2)) == 2
    assert "heldoutword" not in vocab


def test_collate_shapes_and_types():
    ids, lengths, labels = collate_batch([(torch.tensor([2, 3]), torch.tensor(1.0)),
                                         (torch.tensor([4]), torch.tensor(0.0))])
    assert ids.tolist() == [[2, 3], [4, 0]]
    assert lengths.tolist() == [2, 1] and lengths.device.type == "cpu"
    assert labels.dtype == torch.float32 and labels.shape == (2,)


def test_train_only_features(config, bundle):
    bundle.val.loc[0, "text"] = "validationonlyterm"
    features = prepare_tfidf(bundle, config)
    assert hasattr(features.train, "tocsr")
    assert "heldoutword" not in features.vectorizer.vocabulary_
    assert "validationonlyterm" not in features.vectorizer.vocabulary_
    vocab = build_vocab(bundle.train.text, 2000)
    assert "heldoutword" not in vocab
    assert "validationonlyterm" not in vocab
    one = next(iter(make_loaders(bundle, vocab, config)["train"]))
    two = next(iter(make_loaders(bundle, vocab, config)["train"]))
    assert torch.equal(one[0], two[0])
