import argparse
import json
import os
import statistics
import time

import torch

from train_v2 import NanoCNNv2, SmallCNNv2, TinyCNNv2

MODELS = {
    "SmallCNNv2": SmallCNNv2,
    "TinyCNNv2": TinyCNNv2,
    "NanoCNNv2": NanoCNNv2,
}


def load_model(name, path):
    model = MODELS[name]()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def measure(model, repeats=500, warmup=50, rounds=7, threads=1):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, 32, 32)
    samples = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for _ in range(rounds):
            start = time.perf_counter()
            for _ in range(repeats):
                model(x)
            samples.append((time.perf_counter() - start) * 1000 / repeats)
    torch.set_num_threads(old_threads)
    return {
        "median_ms_per_image": statistics.median(samples),
        "mean_ms_per_image": statistics.mean(samples),
        "min_ms_per_image": min(samples),
        "max_ms_per_image": max(samples),
        "rounds": samples,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results_v2")
    p.add_argument("--distill-path", default="results_distill/NanoCNNv2_distilled.pt")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    models = {}
    for name in MODELS:
        path = os.path.join(args.results_dir, f"{name}.pt")
        if os.path.exists(path):
            models[name] = load_model(name, path)

    distill_path = args.distill_path
    if os.path.exists(distill_path):
        models["NanoCNNv2_distilled"] = load_model("NanoCNNv2", distill_path)

    print(f"PyTorch threads during benchmark: {args.threads}")
    output = {}
    for name, model in models.items():
        result = measure(model, threads=args.threads)
        output[name] = result
        print(f"{name}: median={result['median_ms_per_image']:.4f} ms | mean={result['mean_ms_per_image']:.4f} | min={result['min_ms_per_image']:.4f} | max={result['max_ms_per_image']:.4f}")

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "latency_repeated.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {args.results_dir}/latency_repeated.json")

if __name__ == "__main__":
    main()
