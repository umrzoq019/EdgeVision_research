import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    v2 = load_json("results_v2/metrics.json")
    rows = v2["results"]
    names = [r["model"] for r in rows]
    acc = [r["test_accuracy_percent"] for r in rows]
    lat = [r["cpu_latency_ms_per_image"] for r in rows]
    params = [r["trainable_parameters"] for r in rows]

    Path("figures").mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(lat, acc, s=100)
    for i, n in enumerate(names):
        plt.annotate(n, (lat[i], acc[i]), xytext=(8, 8), textcoords="offset points")
    plt.xlabel("CPU latency (ms/image)")
    plt.ylabel("Test accuracy (%)")
    plt.title("EdgeVision v2: Accuracy vs. CPU Efficiency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/accuracy_vs_latency.png", dpi=180)
    plt.show()

    plt.figure(figsize=(8, 6))
    for r in rows:
        plt.plot(range(1, len(r["history"]["val_accuracy"]) + 1), r["history"]["val_accuracy"], label=r["model"])
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy (%)")
    plt.title("EdgeVision v2: Validation Accuracy During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/validation_accuracy.png", dpi=180)
    plt.show()

    print("\nFinal comparison:")
    for n, a, p, l in zip(names, acc, params, lat):
        print(f"{n}: accuracy={a:.2f}% | params={p:,} | latency={l:.3f} ms/image")

    if Path("results_distill/metrics.json").exists():
        dist = load_json("results_distill/metrics.json")["result"]
        print("\nDistilled NanoCNN:")
        print(f"accuracy={dist['test_accuracy_percent']:.2f}% | params={dist['trainable_parameters']:,} | latency={dist['cpu_latency_ms_per_image']:.3f} ms/image")


if __name__ == "__main__":
    main()
