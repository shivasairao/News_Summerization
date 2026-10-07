"""
Data pipeline for CNN/DailyMail news summarization.
Requires: pip install datasets --break-system-packages

Truncates articles to MAX_SRC_LEN tokens and summaries to MAX_TRG_LEN tokens,
builds a word-level vocab capped at VOCAB_SIZE (by frequency), and exposes a
PyTorch Dataset + collate_fn ready for the Seq2Seq model in model.py.
"""

import re
import json
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
MAX_SRC_LEN = 150
MAX_TRG_LEN = 30
VOCAB_SIZE = 8000
N_EXAMPLES = 500_000  # cap; CNN/DailyMail train split has ~287K, so this uses the full train+val+test


def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9.,!?'\s]", " ", text)
    return text.split()


class Vocab:
    def __init__(self, counter, vocab_size):
        specials = [PAD, SOS, EOS, UNK]
        most_common = [w for w, _ in counter.most_common(vocab_size - len(specials))]
        self.itos = specials + most_common
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, tokens, max_len, add_eos=True):
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokens[:max_len - (2 if add_eos else 1)]]
        ids = [self.stoi[SOS]] + ids
        if add_eos:
            ids.append(self.stoi[EOS])
        return ids

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            itos = json.load(f)
        vocab = cls.__new__(cls)
        vocab.itos = itos
        vocab.stoi = {w: i for i, w in enumerate(itos)}
        return vocab


def build_vocab(texts, vocab_size):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    return Vocab(counter, vocab_size)


class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries, vocab):
        self.articles = articles
        self.summaries = summaries
        self.vocab = vocab

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        src_ids = self.vocab.encode(simple_tokenize(self.articles[idx]), MAX_SRC_LEN, add_eos=False)
        trg_ids = self.vocab.encode(simple_tokenize(self.summaries[idx]), MAX_TRG_LEN, add_eos=True)
        return torch.tensor(src_ids), torch.tensor(trg_ids)


def collate_fn(batch, pad_idx):
    srcs, trgs = zip(*batch)
    src_lengths = torch.tensor([len(s) for s in srcs])
    src_padded = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trgs, batch_first=True, padding_value=pad_idx)
    # sort by src_length descending — required by pack_padded_sequence(enforce_sorted=False
    # is actually fine, kept unsorted here on purpose for simplicity)
    return src_padded, src_lengths, trg_padded


def load_cnn_dailymail():
    """
    Loads the CNN/DailyMail 3.0.0 dataset via HuggingFace `datasets`.
    Run this on a machine with internet access (this sandbox has restricted
    network egress and cannot reach huggingface.co).
    """
    from datasets import load_dataset
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0")

    articles, summaries = [], []
    for split in ["train", "validation", "test"]:
        articles.extend(ds[split]["article"])
        summaries.extend(ds[split]["highlights"])
        if len(articles) >= N_EXAMPLES:
            break

    combined = list(zip(articles, summaries))
    random.seed(42)
    random.shuffle(combined)
    combined = combined[:N_EXAMPLES]
    articles, summaries = zip(*combined)
    return list(articles), list(summaries)


def get_dataloaders(batch_size=32, val_frac=0.05, vocab_path="vocab.json"):
    articles, summaries = load_cnn_dailymail()

    import os
    if os.path.exists(vocab_path):
        print(f"Found existing {vocab_path} — reusing it (so RNN/GRU/LSTM, run "
              f"separately, all decode with the exact same vocab).")
        vocab = Vocab.load(vocab_path)
    else:
        vocab = build_vocab(articles[:50_000] + summaries[:50_000], VOCAB_SIZE)  # subset, for speed
        vocab.save(vocab_path)
        print(f"Saved {vocab_path}")

    n_val = int(len(articles) * val_frac)
    train_articles, val_articles = articles[n_val:], articles[:n_val]
    train_summaries, val_summaries = summaries[n_val:], summaries[:n_val]

    train_ds = SummarizationDataset(train_articles, train_summaries, vocab)
    val_ds = SummarizationDataset(val_articles, val_summaries, vocab)

    pad_idx = vocab.stoi[PAD]
    collate = lambda batch: collate_fn(batch, pad_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return train_loader, val_loader, vocab
