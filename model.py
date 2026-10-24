import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import build_dataloader


class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim=100):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, dim)   
        self.out_embed = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.in_embed.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center, context, negatives):
        v = self.in_embed(center)            # (B, D)
        u_pos = self.out_embed(context)      # (B, D)
        u_neg = self.out_embed(negatives)    # (B, K, D)
        pos_loss = F.logsigmoid((v * u_pos).sum(dim=1))
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)
        return -(pos_loss + neg_loss).mean()


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Thiết bị: {device}")
    loader, vocab, neg_dist = build_dataloader(
        args.data, max_words=args.max_words,
        batch_size=args.batch_size, min_count=args.min_count)

    model = SkipGram(len(vocab), args.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        total, steps = 0.0, 0
        for center, context in loader:
            B = center.size(0)
            negatives = torch.multinomial(neg_dist, B * args.neg, replacement=True).view(B, args.neg)
            center, context, negatives = center.to(device), context.to(device), negatives.to(device)
            loss = model(center, context, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps += 1
            if steps % 1000 == 0:
                print(f"  epoch {epoch} | step {steps}/{len(loader)} | loss {total / steps:.4f}", flush=True)
        print(f"Epoch {epoch}/{args.epochs} | loss {total / steps:.4f}")

    torch.save({"embeddings": model.in_embed.weight.detach().cpu(), "itos": vocab.itos}, args.out)
    print(f"Đã lưu embedding vào {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--neg", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--max_words", type=int, default=5_000_000)
    p.add_argument("--min_count", type=int, default=20)
    p.add_argument("--out", default="word2vec.pt")
    train(p.parse_args())