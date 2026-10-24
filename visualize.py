import argparse

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEFAULT_WORDS = ["king", "queen", "man", "woman", "france",
                 "paris", "italy", "rome", "three", "four"]


def main(args):
    ckpt = torch.load(args.emb, map_location="cpu")
    emb, stoi = ckpt["embeddings"], {w: i for i, w in enumerate(ckpt["itos"])}

    words = [w for w in args.words if w in stoi]
    for w in args.words:
        if w not in stoi:
            print(f"Bỏ qua '{w}': không có trong từ điển")

    vectors = emb[[stoi[w] for w in words]].numpy()
    if args.method == "pca":
        xy = PCA(n_components=2).fit_transform(vectors)
    else:
        xy = TSNE(n_components=2, perplexity=min(5, len(words) - 1),
                  init="pca", random_state=0).fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    plt.scatter(xy[:, 0], xy[:, 1], s=40)
    for (x, y), w in zip(xy, words):
        plt.annotate(w, (x, y), fontsize=12, xytext=(5, 5), textcoords="offset points")
    plt.title(f"Word embedding ({args.method.upper()}) - {len(words)} từ")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Đã lưu {args.out}")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="word2vec.pt")
    p.add_argument("--words", nargs="*", default=DEFAULT_WORDS)
    p.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    p.add_argument("--out", default="embedding.png")
    main(p.parse_args())