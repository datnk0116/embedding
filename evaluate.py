import argparse

import torch
import torch.nn.functional as F

ANALOGIES = [
    ("man", "king", "woman", "queen"),
    ("father", "mother", "king", "queen"),
    ("brother", "sister", "son", "daughter"),
    ("he", "she", "his", "her"),
    ("france", "paris", "italy", "rome"),
    ("france", "paris", "germany", "berlin"),
    ("france", "paris", "england", "london"),
    ("russia", "moscow", "france", "paris"),
    ("japan", "tokyo", "china", "beijing"),
    ("france", "french", "germany", "german"),
    ("france", "french", "italy", "italian"),
    ("one", "two", "three", "four"),
    ("january", "february", "march", "april"),
    ("big", "bigger", "small", "smaller"),
    ("long", "longer", "short", "shorter"),
    ("good", "better", "bad", "worse"),
    ("walk", "walked", "go", "went"),
    ("usa", "dollar", "japan", "yen"),
    ("sun", "day", "moon", "night"),
]

DEFAULT_WORDS = ["king", "france", "computer", "war", "three", "music", "water", "city"]


def load(path):
    ckpt = torch.load(path, map_location="cpu")
    emb = F.normalize(ckpt["embeddings"], dim=1)  
    itos = ckpt["itos"]
    stoi = {w: i for i, w in enumerate(itos)}
    return emb, itos, stoi


def most_similar(emb, itos, stoi, word, k=8):
    sims = emb @ emb[stoi[word]]
    top = sims.topk(k + 1).indices.tolist()
    return [(itos[i], sims[i].item()) for i in top if itos[i] != word][:k]


def solve_analogy(emb, itos, stoi, a, b, c, k=5):
    v = emb[stoi[b]] - emb[stoi[a]] + emb[stoi[c]]
    sims = emb @ F.normalize(v, dim=0)
    for w in (a, b, c):  # không cho phép trả lại chính các từ đầu vào
        sims[stoi[w]] = -1e9
    return [itos[i] for i in sims.topk(k).indices.tolist()]


def main(args):
    emb, itos, stoi = load(args.emb)
    print(f"Từ điển: {len(itos):,} từ | số chiều: {emb.size(1)}\n")

    print("=== Từ gần nghĩa nhất (cosine similarity) ===")
    for w in args.words:
        if w not in stoi:
            print(f"{w}: không có trong từ điển")
            continue
        near = most_similar(emb, itos, stoi, w, args.k)
        print(f"{w:>10s}: " + ", ".join(f"{x} ({s:.2f})" for x, s in near))

    print("\n=== Phép toán từ  a : b :: c : ?  (dự đoán = b - a + c) ===")
    top1 = top5 = total = 0
    for a, b, c, d in ANALOGIES:
        if any(w not in stoi for w in (a, b, c, d)):
            print(f"Bỏ qua ({a}, {b}, {c}, {d}): thiếu từ trong từ điển")
            continue
        preds = solve_analogy(emb, itos, stoi, a, b, c, 5)
        total += 1
        top1 += preds[0] == d
        top5 += d in preds
        mark = "OK  " if preds[0] == d else ("top5" if d in preds else "sai ")
        print(f"[{mark}] {a}:{b} :: {c}:?  ->  {preds}   (đáp án: {d})")

    print(f"\nĐộ chính xác top-1: {top1}/{total} = {top1 / total:.0%}")
    print(f"Độ chính xác top-5: {top5}/{total} = {top5 / total:.0%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="word2vec.pt")
    p.add_argument("--words", nargs="*", default=DEFAULT_WORDS)
    p.add_argument("--k", type=int, default=8)
    main(p.parse_args())