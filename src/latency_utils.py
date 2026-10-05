"""Shared CPU-latency helpers.

Why this exists: the original benchmarks timed each model in its own block, one
after another. CPU frequency, noisy neighbours (Colab) and thermal state drift
over minutes, so a model measured later can look 30-60% faster or slower purely
by timing. Here every round times *all* models back-to-back in a random order,
so drift hits all models alike, and ratios can be computed per round.
"""
import os
import platform
import random
import statistics
import time

import torch


def environment_info() -> dict:
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu,
        "logical_cpus": os.cpu_count(),
        "torch_threads_default": torch.get_num_threads(),
    }


def _quartiles(values):
    s = sorted(values)
    n = len(s)

    def q(p):
        k = (n - 1) * p
        lo = int(k)
        hi = min(lo + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return q(0.25), q(0.75)


def interleaved_latency(models: dict, rounds: int = 30, repeats: int = 200,
                        warmup: int = 100, threads: int = 1, seed: int = 0,
                        input_shape=(1, 3, 32, 32)) -> dict:
    """Time every model in `models` (name -> callable/eval module) in each round.

    Returns per-model statistics in ms/image, including the raw per-round list
    (round i of every model was measured in the same short time window).
    """
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    names = list(models)
    x = torch.randn(*input_shape)
    samples = {n: [] for n in names}
    try:
        with torch.inference_mode():
            for n in names:
                models[n].eval()
                for _ in range(warmup):
                    models[n](x)
            for r in range(rounds):
                order = names[:]
                random.Random(seed + r).shuffle(order)
                for n in order:
                    m = models[n]
                    t0 = time.perf_counter()
                    for _ in range(repeats):
                        m(x)
                    samples[n].append((time.perf_counter() - t0) * 1000.0 / repeats)
    finally:
        torch.set_num_threads(previous_threads)

    out = {}
    for n, v in samples.items():
        q1, q3 = _quartiles(v)
        out[n] = {
            "median_ms_per_image": statistics.median(v),
            "mean_ms_per_image": statistics.mean(v),
            "std_ms_per_image": statistics.stdev(v) if len(v) > 1 else 0.0,
            "p25_ms_per_image": q1,
            "p75_ms_per_image": q3,
            "min_ms_per_image": min(v),
            "max_ms_per_image": max(v),
            "rounds": v,
        }
    return out


def paired_round_ratio(result: dict, numerator: str, denominator: str) -> dict:
    """Median of per-round ratios numerator/denominator (robust to drift)."""
    ratios = [a / b for a, b in zip(result[numerator]["rounds"], result[denominator]["rounds"])]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "median_ratio": statistics.median(ratios),
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
    }
