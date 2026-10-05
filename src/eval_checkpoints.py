"""Evaluate saved FP32 checkpoints on the FULL CIFAR-10 test set (10,000 images).

The v2 results use only the first 1,000 test images (binomial standard error of
about 1.5 percentage points). Evaluating checkpoints on all 10,000 images needs
no retraining, shrinks that to ~0.5 pp, and the saved per-image predictions allow
paired tests (McNemar) between models.

    python src/eval_checkpoints.py \
        --checkpoints results/multi_seed/seed_42/NanoCNNv2.pt results/distillation/multi_seed/seed_42/NanoCNNv2_distilled.pt \
        --out results/analysis/full_test_eval.json

Architecture is inferred from the file name (Small/Tiny/Nano; '_distilled' -> Nano).
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_v2 import CIFAR10_MEAN, CIFAR10_STD, MODELS


def arch_of(path: Path) -> str:
    for name in MODELS:
        if path.stem.startswith(name):
            return name
    raise ValueError(f"cannot infer architecture from {path.name}")


def wilson(correct, n, z=1.96):
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * (centre - half), 100 * (centre + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--out", default="results/analysis/full_test_eval.json")
    args = ap.parse_args()

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=tfm)
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=2)
    labels = np.array(test.targets)

    out, preds_store = {}, {}
    for ck in args.checkpoints:
        path = Path(ck)
        model = MODELS[arch_of(path)]()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        preds = []
        with torch.inference_mode():
            for images, _ in loader:
                preds.append(model(images).argmax(1).numpy())
        preds = np.concatenate(preds)
        correct = int((preds == labels).sum())
        lo, hi = wilson(correct, len(labels))
        key = str(path)
        out[key] = {"architecture": arch_of(path), "n": len(labels), "correct": correct,
                    "test_accuracy_percent": 100 * correct / len(labels), "wilson95_percent": [lo, hi]}
        preds_store[key] = preds
        print(f"{path.name}: {out[key]['test_accuracy_percent']:.2f}% (95% CI {lo:.2f}-{hi:.2f})")

    # McNemar exact-ish (chi-square with continuity) for every pair
    keys = list(preds_store)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = preds_store[keys[i]] == labels, preds_store[keys[j]] == labels
            n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
            chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) else 0.0
            p = math.erfc(math.sqrt(chi2 / 2))  # chi-square, 1 dof
            pairs.append({"a": keys[i], "b": keys[j], "only_b_correct": n01, "only_a_correct": n10,
                          "mcnemar_chi2": chi2, "p_value": p})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"models": out, "mcnemar": pairs}, f, indent=2)
    np.savez_compressed(os.path.splitext(args.out)[0] + "_predictions.npz", labels=labels,
                        **{Path(k).parent.name + "__" + Path(k).stem: v for k, v in preds_store.items()})
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
