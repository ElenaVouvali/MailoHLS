#!/usr/bin/env python3
"""Train the additive complete-design ranker from tensorized pair records.

Each JSONL row contains ``chosen`` and ``rejected`` objects with lists named
``structural``, ``budget``, ``directive``, ``stage2_logprob`` and ``qor``.
This keeps feature extraction (which is model/tokenizer-specific) explicit.
"""
import argparse, json, os
import torch
from torch.utils.data import DataLoader
from LLM_branch.train.whole_design_reranker import WholeDesignRanker, RerankerDimensions, whole_design_pair_loss

def tensor(x, device): return torch.tensor(x, dtype=torch.float32, device=device)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True); ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64); ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args(); torch.manual_seed(a.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [json.loads(x) for x in open(a.pairs, encoding="utf-8") if x.strip()]
    if not rows: raise ValueError("empty pair file")
    sample = rows[0]["chosen"]
    dims = RerankerDimensions(len(sample["structural"]), len(sample["budget"]), len(sample["directive"]), hidden=a.hidden)
    model = WholeDesignRanker(dims).to(device); opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    for _ in range(a.epochs):
        for start in range(0, len(rows), a.batch_size):
            batch = rows[start:start+a.batch_size]
            def pack(side):
                return (torch.stack([tensor(r[side]["structural"], device) for r in batch]),
                        torch.stack([tensor(r[side]["budget"], device) for r in batch]),
                        torch.stack([tensor(r[side]["directive"], device) for r in batch]),
                        tensor([r[side]["stage2_logprob"] for r in batch], device),
                        tensor([r[side]["qor"] for r in batch], device))
            c = pack("chosen"); r = pack("rejected")
            sc = model(*c[:4]); sr = model(*r[:4]); gaps = torch.log(r[4] / c[4]).clamp_min(0)
            med = max(float(gaps.detach().median().item()), 1e-8)
            loss = whole_design_pair_loss(sc, sr, a.temperature, c[4], r[4], med)
            opt.zero_grad(); loss.backward(); opt.step()
    os.makedirs(a.output_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "dims": dims.__dict__}, os.path.join(a.output_dir, "ranker.pt"))
    with open(os.path.join(a.output_dir, "ranker_config.json"), "w") as f: json.dump({"dims": dims.__dict__, "temperature": a.temperature}, f, indent=2)
    print(f"saved {a.output_dir}/ranker.pt ({len(rows)} pairs)")
if __name__ == "__main__": main()
