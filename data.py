import zipfile
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def read_text8(path, max_words=None):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            text = z.read(z.namelist()[0]).decode("utf-8")
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    words = text.split()
    return words[:max_words] if max_words is not None else words


class Vocab:
    def __init__(self, words, min_count=5):
        counter = Counter(words)
        items = [(w, c) for w, c in counter.items() if c >= min_count]
        items.sort(key=lambda x: -x[1])
        self.itos = [w for w, _ in items]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = np.array([c for _, c in items], dtype=np.float64)

    def __len__(self):
        return len(self.itos)

    def encode(self, words):
        return np.array([self.stoi[w] for w in words if w in self.stoi], dtype=np.int64)

    def neg_distribution(self, power=0.75):
        probs = self.counts ** power
        probs /= probs.sum()
        return torch.tensor(probs, dtype=torch.float)


def make_pairs(tokens, window=5, seed=0):
    rng = np.random.default_rng(seed)
    n = len(tokens)
    win_sizes = rng.integers(1, window + 1, size=n)
    centers, contexts = [], []
    for off in range(1, window + 1):
        idx = np.arange(n - off)              # context bên phải
        ok = win_sizes[idx] >= off
        centers.append(tokens[idx[ok]])
        contexts.append(tokens[idx[ok] + off])
        idx = np.arange(off, n)               # context bên trái
        ok = win_sizes[idx] >= off
        centers.append(tokens[idx[ok]])
        contexts.append(tokens[idx[ok] - off])
    return np.concatenate(centers), np.concatenate(contexts)


def build_dataloader(path, max_words=5_000_000, min_count=5, window=5, batch_size=4096):
    words = read_text8(path, max_words)
    vocab = Vocab(words, min_count)
    tokens = vocab.encode(words)
    centers, contexts = make_pairs(tokens, window)
    print(f"Số từ: {len(words):,} | từ vựng: {len(vocab):,} | cặp huấn luyện: {len(centers):,}")
    dataset = TensorDataset(torch.from_numpy(centers), torch.from_numpy(contexts))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, vocab, vocab.neg_distribution()