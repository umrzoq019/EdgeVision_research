
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(
    seed,
    models,
    epochs,
    total_train,
    val_size,
    test_subset,
    batch_size,
    lr,
    weight_decay,
    data_dir,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train_v2.py"),
        "--models",
        models,
        "--epochs",
        str(epochs),
        "--total-train",
        str(total_train),
        "--val-size",
        str(val_size),
        "--test-subset",
        str(test_subset),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(output_dir),
        "--cpu-threads",
        "1",
        "--latency-repeats",
        "100",
        "--latency-warmup",
        "20",
        "--seed",
        str(seed),
    ]

    print("\n" + "=" * 80)
    print(f"RUNNING SEED {seed}")
    print("=" * 80)
    print(" ".join(command))

    subprocess.run(command, check=True)

    metrics_path = output_dir / "metrics.json"

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(all_results):
    grouped = {}

    for seed_result in all_results:
        seed = seed_result["seed"]

        for result in seed_result["results"]:
            model = result["model"]

            grouped.setdefault(model, []).append({
                "seed": seed,
                "test_accuracy_percent":
                    result["test_accuracy_percent"],
                "best_validation_accuracy_percent":
                    result["best_validation_accuracy_percent"],
                "trainable_parameters":
                    result["trainable_parameters"],
                "model_size_mb":
                    result["model_size_mb"],
                "cpu_latency_ms_per_image":
                    result["cpu_latency_ms_per_image"],
            })

    summary = {}

    for model, values in grouped.items():
        accuracies = np.array([
            x["test_accuracy_percent"] for x in values
        ])

        val_accuracies = np.array([
            x["best_validation_accuracy_percent"] for x in values
        ])

        latencies = np.array([
            x["cpu_latency_ms_per_image"] for x in values
        ])

        params = [x["trainable_parameters"] for x in values]
        sizes = [x["model_size_mb"] for x in values]

        summary[model] = {
            "seeds": [x["seed"] for x in values],

            "test_accuracy_percent": {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies, ddof=1)),
                "values": accuracies.tolist(),
            },

            "best_validation_accuracy_percent": {
                "mean": float(np.mean(val_accuracies)),
                "std": float(np.std(val_accuracies, ddof=1)),
                "values": val_accuracies.tolist(),
            },

            "cpu_latency_ms_per_image": {
                "mean": float(np.mean(latencies)),
                "std": float(np.std(latencies, ddof=1)),
                "values": latencies.tolist(),
            },

            "trainable_parameters": params,
            "model_size_mb": sizes,
        }

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
    )

    parser.add_argument(
        "--models",
        default="SmallCNNv2,TinyCNNv2,NanoCNNv2",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--total-train", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-subset", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument(
        "--data-dir",
        default="data",
    )

    parser.add_argument(
        "--output-root",
        default="results/multi_seed",
    )

    args = parser.parse_args()

    set_seed(42)

    all_results = []

    for seed in args.seeds:
        seed_dir = Path(args.output_root) / f"seed_{seed}"

        result = run_training(
            seed=seed,
            models=args.models,
            epochs=args.epochs,
            total_train=args.total_train,
            val_size=args.val_size,
            test_subset=args.test_subset,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            data_dir=args.data_dir,
            output_dir=seed_dir,
        )

        all_results.append(result)

        # Save progress immediately after each seed.
        progress_path = Path(args.output_root) / "raw_results.json"

        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nSaved progress: {progress_path}")

    summary = summarize(all_results)

    summary_payload = {
        "experiment": "EdgeVision v2 multi-seed validation",
        "seeds": args.seeds,
        "models": [
            x.strip()
            for x in args.models.split(",")
            if x.strip()
        ],
        "settings": {
            "epochs": args.epochs,
            "total_train": args.total_train,
            "val_size": args.val_size,
            "test_subset": args.test_subset,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
        },
        "summary": summary,
    }

    summary_path = Path(args.output_root) / "summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY")
    print("=" * 80)

    for model, values in summary.items():
        acc = values["test_accuracy_percent"]
        lat = values["cpu_latency_ms_per_image"]

        print(
            f"{model}: "
            f"accuracy={acc['mean']:.2f} ± {acc['std']:.2f}% | "
            f"latency={lat['mean']:.4f} ± {lat['std']:.4f} ms"
        )

    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
