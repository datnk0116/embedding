"""data.py - chuẩn bị dữ liệu Skip-gram từ text8.

Quy trình: đọc text -> xây từ điển -> mã hóa thành index -> subsampling
-> tạo cặp (từ trung tâm, từ ngữ cảnh) -> DataLoader.
"""
import os
import zipfile
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def read_text8(path: str, max_words: int | None = None) -> list[str]:
    """Đọc text8 (file .zip hoặc file text thường) và trả về list các từ."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = z.namelist()[0]
            text = z.read(name).decode("utf-8")
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    words = text.split()
    if max_words is not None:
        words = words[:max_words]
    return words


class Vocab:
    """Từ điển: word <-> index, kèm tần suất và phân phối lấy mẫu âm."""

    def __init__(self, words: list[str], min_count: int = 5):
        counter = Counter(words)
        # Chỉ giữ từ xuất hiện >= min_count, sắp xếp theo tần suất giảm dần
        items = [(w, c) for w, c in counter.items() if c >= min_count]
        items.sort(key=lambda x: -x[1])

        self.itos = [w for w, _ in items]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = np.array([c for _, c in items], dtype=np.float64)
        self.total = self.counts.sum()

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, words: list[str]) -> np.ndarray:
        """Đổi list từ thành mảng index, bỏ từ không có trong từ điển."""
        return np.array([self.stoi[w] for w in words if w in self.stoi], dtype=np.int64)

    def neg_distribution(self, power: float = 0.75) -> torch.Tensor:
        """Phân phối lấy mẫu âm: tần suất ^ 0.75 (như bài báo gốc)."""
        probs = self.counts ** power
        probs /= probs.sum()
        return torch.tensor(probs, dtype=torch.float)


def subsample(tokens: np.ndarray, vocab: Vocab, t: float = 1e-5,
              seed: int = 0) -> np.ndarray:
    """Bỏ ngẫu nhiên các từ quá phổ biến ("the", "of", ...).

    Xác suất GIỮ từ w: min(1, sqrt(t / f(w)))  với f(w) là tần suất tương đối.
    """
    rng = np.random.default_rng(seed)
    freq = vocab.counts / vocab.total
    keep_prob = np.minimum(1.0, np.sqrt(t / freq))
    mask = rng.random(len(tokens)) < keep_prob[tokens]
    return tokens[mask]


def make_pairs(tokens: np.ndarray, window: int = 5, seed: int = 0):
    """Tạo cặp (center, context) cho Skip-gram.

    Với mỗi vị trí, dùng kích thước cửa sổ ngẫu nhiên từ 1..window
    (từ gần được lấy nhiều hơn từ xa, giống bản gốc).
    """
    rng = np.random.default_rng(seed)
    n = len(tokens)
    win_sizes = rng.integers(1, window + 1, size=n)
    centers, contexts = [], []

    for off in range(1, window + 1):
        # Vị trí i có dùng offset này không: win_sizes[i] >= off
        # Context bên phải: i + off
        idx = np.arange(n - off)
        ok = win_sizes[idx] >= off
        centers.append(tokens[idx[ok]])
        contexts.append(tokens[idx[ok] + off])
        # Context bên trái: i - off
        idx = np.arange(off, n)
        ok = win_sizes[idx] >= off
        centers.append(tokens[idx[ok]])
        contexts.append(tokens[idx[ok] - off])

    return np.concatenate(centers), np.concatenate(contexts)


def build_dataloader(path: str, max_words: int | None = 5_000_000,
                     min_count: int = 5, window: int = 5,
                     subsample_t: float = 1e-5, batch_size: int = 4096,
                     num_workers: int = 0):
    """Hàm chính: trả về (dataloader, vocab, neg_dist)."""
    print(f"Đọc dữ liệu từ {path} ...")
    words = read_text8(path, max_words)
    print(f"  Số từ: {len(words):,}")

    vocab = Vocab(words, min_count)
    print(f"  Kích thước từ điển (min_count={min_count}): {len(vocab):,}")

    tokens = vocab.encode(words)
    tokens = subsample(tokens, vocab, subsample_t)
    print(f"  Số từ sau subsampling: {len(tokens):,}")

    centers, contexts = make_pairs(tokens, window)
    print(f"  Số cặp huấn luyện: {len(centers):,}")

    dataset = TensorDataset(torch.from_numpy(centers), torch.from_numpy(contexts))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True)
    return loader, vocab, vocab.neg_distribution()


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "text8.zip"
    if not os.path.exists(path):
        raise SystemExit(f"Không tìm thấy file: {path}\nDùng: python data.py <đường_dẫn_text8>")

    loader, vocab, neg_dist = build_dataloader(path)
    center, context = next(iter(loader))
    print("\nVí dụ 5 cặp đầu của batch:")
    for c, x in zip(center[:5].tolist(), context[:5].tolist()):
        print(f"  {vocab.itos[c]:>12s} -> {vocab.itos[x]}")
    print("Top 10 từ phổ biến:", vocab.itos[:10])