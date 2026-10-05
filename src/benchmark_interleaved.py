"""Interleaved single-thread CPU latency benchmark for the three FP32 architectures.

Latency of a dense CNN does not depend on the trained weights, so checkpoints are
optional: without --checkpoint-root the models are randomly initialised.
NanoCNNv2 and NanoCNNv2_distilled share one architecture, so they are one entry.

    python src/benchmark_repeated.py            # legacy, sequential
    python src/benchmark_interleaved.py         # recommended
"""
import argparse
import json
import os
from pathlib import Path

import torch

from latency_utils import environment_info, interleaved_latency, paired_round_ratio
from train_v2 import MODELS, set_seed


def build(name, checkpoint_root, seed):
    model = MODELS[name]()
    if checkpoint_root:
        path = Path(checkpoint_root) / f"seed_{seed}" / f"{name}.pt"
        if path.exists():
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        else:
            print(f"[warn] {path} not found; using random init for {name}")
    return model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", default=None, help="e.g. results/multi_seed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default="results/latency/interleaved.json")
    args = p.parse_args()

    set_seed(args.seed)
    models = {name: build(name, args.checkpoint_root, args.seed) for name in MODELS}
    result = interleaved_latency(models, args.rounds, args.repeats, args.warmup, args.threads, args.seed)

    ratios = {
        f"{name}_vs_SmallCNNv2": paired_round_ratio(result, name, "SmallCNNv2")
        for name in MODELS if name != "SmallCNNv2"
    }
    for name, r in result.items():
        print(f"{name}: median={r['median_ms_per_image']:.4f} ms | IQR=[{r['p25_ms_per_image']:.4f}, "
              f"{r['p75_ms_per_image']:.4f}] | min={r['min_ms_per_image']:.4f} | max={r['max_ms_per_image']:.4f}")
    for k, v in ratios.items():
        print(f"{k}: median per-round latency ratio = {v['median_ratio']:.3f}")

    payload = {
        "protocol": {"rounds": args.rounds, "repeats_per_round": args.repeats, "warmup": args.warmup,
                     "threads": args.threads, "order": "randomised per round", "weights": (
                         "checkpoints" if args.checkpoint_root else "random init")},
        "environment": environment_info(),
        "models": result,
        "per_round_ratios_vs_small": ratios,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
