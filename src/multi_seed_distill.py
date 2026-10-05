"""Run knowledge distillation (SmallCNNv2 -> NanoCNNv2) for several seeds.

Each seed uses the teacher trained with the same seed by src/multi_seed.py and
calls src/distill_v2.py exactly as the preserved results/distillation/multi_seed
metrics were produced (T=2.0, alpha=0.5 by default). If baseline per-seed
metrics exist, paired differences (distilled - baseline NanoCNNv2) are added.

    python src/multi_seed.py --seeds 42 123 456
    python src/multi_seed_distill.py --seeds 42 123 456
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--baseline-root", default="results/multi_seed")
    p.add_argument("--output-root", default="results/distillation/multi_seed")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--total-train", type=int, default=10000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--test-subset", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    script = Path(__file__).resolve().parent / "distill_v2.py"
    results = []
    for seed in args.seeds:
        out_dir = Path(args.output_root) / f"seed_{seed}"
        cmd = [sys.executable, str(script),
               "--teacher", str(Path(args.baseline_root) / f"seed_{seed}" / "SmallCNNv2.pt"),
               "--out-dir", str(out_dir), "--temperature", str(args.temperature), "--alpha", str(args.alpha),
               "--epochs", str(args.epochs), "--total-train", str(args.total_train),
               "--val-size", str(args.val_size), "--test-subset", str(args.test_subset),
               "--batch-size", str(args.batch_size), "--lr", str(args.lr),
               "--weight-decay", str(args.weight_decay), "--data-dir", args.data_dir,
               "--cpu-threads", "1", "--latency-repeats", "100", "--latency-warmup", "20",
               "--seed", str(seed)]
        print("\n" + "=" * 80 + f"\nDISTILL SEED {seed}\n" + "=" * 80)
        subprocess.run(cmd, check=True)
        with open(out_dir / "metrics.json", encoding="utf-8") as f:
            results.append((seed, json.load(f)["result"]))

    def stat(key):
        v = np.array([r[key] for _, r in results], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()}

    summary = {
        "experiment": "EdgeVision v2 multi-seed knowledge distillation",
        "seeds": args.seeds, "student": "NanoCNNv2", "teacher": "SmallCNNv2",
        "settings": {"epochs": args.epochs, "total_train": args.total_train, "val_size": args.val_size,
                     "test_subset": args.test_subset, "batch_size": args.batch_size,
                     "learning_rate": args.lr, "weight_decay": args.weight_decay,
                     "temperature": args.temperature, "alpha": args.alpha},
        "results": {k: stat(k) for k in ("test_accuracy_percent", "best_validation_accuracy_percent",
                                         "cpu_latency_ms_per_image")},
    }

    base = []
    for seed, _ in results:
        mp = Path(args.baseline_root) / f"seed_{seed}" / "metrics.json"
        if not mp.exists():
            base = []
            break
        with open(mp, encoding="utf-8") as f:
            nano = next(r for r in json.load(f)["results"] if r["model"] == "NanoCNNv2")
        base.append(nano["test_accuracy_percent"])
    if base:
        diff = np.array([r["test_accuracy_percent"] for _, r in results]) - np.array(base)
        summary["paired_vs_baseline_nano"] = {
            "baseline_test_accuracy_percent": base, "difference_pp": diff.tolist(),
            "mean_difference_pp": float(diff.mean()),
            "sd_difference_pp": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0}

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_root) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {Path(args.output_root) / 'summary.json'}")


if __name__ == "__main__":
    main()
