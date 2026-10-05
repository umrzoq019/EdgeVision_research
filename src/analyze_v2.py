"""Single-seed figures (accuracy vs latency, validation curves).

For the full multi-seed analysis use src/final_analysis.py.
Reads a train_v2.py / multi_seed.py metrics.json that includes per-epoch history.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/multi_seed/seed_42/metrics.json")
    ap.add_argument("--distill", default="results/distillation/multi_seed/seed_42/metrics.json")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args()

    rows = load_json(args.metrics)["results"]
    names = [r["model"] for r in rows]
    acc = [r["test_accuracy_percent"] for r in rows]
    lat = [r["cpu_latency_ms_per_image"] for r in rows]
    Path(args.fig_dir).mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(lat, acc, s=100)
    for i, n in enumerate(names):
        plt.annotate(n, (lat[i], acc[i]), xytext=(8, 8), textcoords="offset points")
    plt.xlabel("CPU latency (ms/image, single noisy measurement at end of training)")
    plt.ylabel("Test accuracy (%)")
    plt.title("EdgeVision v2 (one seed): Accuracy vs. CPU latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(args.fig_dir) / "single_seed_accuracy_vs_latency.png", dpi=180)
    plt.close()

    if all("history" in r for r in rows):
        plt.figure(figsize=(8, 6))
        for r in rows:
            h = r["history"]["val_accuracy"]
            plt.plot(range(1, len(h) + 1), h, label=r["model"])
        plt.xlabel("Epoch")
        plt.ylabel("Validation accuracy (%)")
        plt.title("EdgeVision v2 (one seed): validation accuracy during training")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(args.fig_dir) / "validation_accuracy.png", dpi=180)
        plt.close()

    print("\nFinal comparison:")
    for r in rows:
        print(f"{r['model']}: accuracy={r['test_accuracy_percent']:.2f}% | params={r['trainable_parameters']:,} "
              f"| latency={r['cpu_latency_ms_per_image']:.3f} ms/image")

    if Path(args.distill).exists():
        d = load_json(args.distill)["result"]
        print(f"\nDistilled NanoCNN: accuracy={d['test_accuracy_percent']:.2f}% | params={d['trainable_parameters']:,}")


if __name__ == "__main__":
    main()
