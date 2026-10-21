"""visualize.py - trực quan hóa word embedding bằng t-SNE hoặc PCA.

Chế độ 1 (mặc định): vẽ N từ phổ biến nhất (bỏ các từ chức năng như "the", "of").
    python3 visualize.py
Chế độ 2: vẽ các từ chọn trước cùng các từ gần nghĩa nhất, tô màu theo nhóm.
    python3 visualize.py --words king france computer war three
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load(path):
    ckpt = torch.load(path, map_location="cpu")
    emb = ckpt["embeddings"]
    itos = ckpt["itos"]
    stoi = {w: i for i, w in enumerate(itos)}
    return emb, itos, stoi


def neighbors(emb, stoi, itos, word, k):
    emb_norm = F.normalize(emb, dim=1)
    sims = emb_norm @ emb_norm[stoi[word]]
    top = sims.topk(k + 1).indices.tolist()
    return [itos[i] for i in top if itos[i] != word][:k]


def reduce_2d(vectors, method):
    if method == "pca":
        return PCA(n_components=2).fit_transform(vectors)
    perplexity = min(30, max(5, len(vectors) // 4))
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=0).fit_transform(vectors)


def plot_top_words(emb, itos, n, skip, method, out):
    idx = list(range(skip, skip + n))
    words = [itos[i] for i in idx]
    xy = reduce_2d(emb[idx].numpy(), method)

    plt.figure(figsize=(16, 12))
    plt.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.5)
    for (x, y), w in zip(xy, words):
        plt.annotate(w, (x, y), fontsize=8, alpha=0.8)
    plt.title(f"Word embedding ({method.upper()}) - {n} từ phổ biến (bỏ {skip} từ đầu)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Đã lưu {out}")
    plt.show()


def plot_groups(emb, itos, stoi, seeds, k, method, out):
    words, groups = [], []
    for g, seed in enumerate(seeds):
        if seed not in stoi:
            print(f"Bỏ qua '{seed}': không có trong từ điển")
            continue
        for w in [seed] + neighbors(emb, stoi, itos, seed, k):
            if w not in words:  # mỗi từ chỉ thuộc nhóm đầu tiên chứa nó
                words.append(w)
                groups.append(g)
    if not words:
        raise SystemExit("Không có từ nào hợp lệ để vẽ.")

    vectors = emb[[stoi[w] for w in words]].numpy()
    xy = reduce_2d(vectors, method)
    groups = np.array(groups)
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(16, 12))
    for g, seed in enumerate(seeds):
        mask = groups == g
        if mask.any():
            plt.scatter(xy[mask, 0], xy[mask, 1], s=30, color=cmap(g % 10), label=seed)
    for (x, y), w in zip(xy, words):
        is_seed = w in seeds
        plt.annotate(w, (x, y), fontsize=11 if is_seed else 8,
                     fontweight="bold" if is_seed else "normal")
    plt.legend(title="Từ gốc")
    plt.title(f"Word embedding ({method.upper()}) - từ gốc và {k} từ gần nhất")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Đã lưu {out}")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="word2vec.pt")
    p.add_argument("--words", nargs="*", help="các từ gốc để vẽ theo nhóm")
    p.add_argument("--k", type=int, default=15, help="số từ gần nhất mỗi nhóm")
    p.add_argument("--n", type=int, default=300, help="số từ ở chế độ 1")
    p.add_argument("--skip", type=int, default=50, help="bỏ bao nhiêu từ phổ biến nhất")
    p.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    p.add_argument("--out", default="embedding.png")
    a = p.parse_args()

    emb, itos, stoi = load(a.emb)
    if a.words:
        plot_groups(emb, itos, stoi, a.words, a.k, a.method, a.out)
    else:
        plot_top_words(emb, itos, a.n, a.skip, a.method, a.out)