"""Post-training static INT8 quantization (PyTorch eager mode) + interleaved latency.

Reconstructed to match the protocol recorded in results/quantization/metrics.json
(x86 engine, 1000 calibration images, 1000 test images, 1 thread). The original
Colab code was not preserved, so this is a re-implementation, not an export.
Calibration uses the *validation split* (clean, non-augmented), never the test set.

    python src/quantize_ptq.py --checkpoint results/multi_seed/seed_42/NanoCNNv2.pt \
        --model NanoCNNv2 --out-dir results/quantization_rerun
"""
import argparse
import json
import os

import torch
import torch.ao.quantization as tq
from torch import nn

from latency_utils import environment_info, interleaved_latency, paired_round_ratio
from train_v2 import MODELS, SEED, accuracy, make_loaders, set_seed


class QuantWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)


def fusion_groups(model):
    """Conv2d -> BatchNorm2d -> ReLU triples inside model.features."""
    children = list(model.features.named_children())
    groups = []
    for i in range(len(children) - 2):
        (n0, m0), (n1, m1), (n2, m2) = children[i:i + 3]
        if isinstance(m0, nn.Conv2d) and isinstance(m1, nn.BatchNorm2d) and isinstance(m2, nn.ReLU):
            groups.append([f"model.features.{n0}", f"model.features.{n1}", f"model.features.{n2}"])
    return groups


def file_size_mb(state_dict, path):
    torch.save(state_dict, path)
    return os.path.getsize(path) / (1024 * 1024)


def fp32_payload_bytes(state_dict):
    return sum(t.numel() * t.element_size() for t in state_dict.values() if torch.is_tensor(t))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="NanoCNNv2", choices=list(MODELS))
    p.add_argument("--out-dir", default="results/quantization_rerun")
    p.add_argument("--engine", default="x86", help="x86 (torch>=2.0), fbgemm, qnnpack")
    p.add_argument("--calib-samples", type=int, default=1000)
    p.add_argument("--total-train", type=int, default=10000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--test-subset", type=int, default=1000, help="use 10000 for the full CIFAR-10 test set")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    torch.backends.quantized.engine = args.engine
    cpu = torch.device("cpu")

    _, val_loader, test_loader = make_loaders(
        args.data_dir, args.total_train, args.val_size, args.test_subset, args.batch_size, args.seed)

    fp32 = MODELS[args.model]()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    fp32.load_state_dict(state)
    fp32.eval()

    wrapped = QuantWrapper(MODELS[args.model]())
    wrapped.model.load_state_dict(state)
    wrapped.eval()
    tq.fuse_modules(wrapped, fusion_groups(wrapped.model), inplace=True)
    wrapped.qconfig = tq.get_default_qconfig(args.engine)
    prepared = tq.prepare(wrapped, inplace=False)

    seen = 0
    with torch.inference_mode():
        for images, _ in val_loader:  # clean (non-augmented) validation images
            prepared(images)
            seen += images.size(0)
            if seen >= args.calib_samples:
                break
    int8 = tq.convert(prepared, inplace=False).eval()

    fp32_val, fp32_test = accuracy(fp32, val_loader, cpu), accuracy(fp32, test_loader, cpu)
    int8_val, int8_test = accuracy(int8, val_loader, cpu), accuracy(int8, test_loader, cpu)
    fp32_mb = file_size_mb(fp32.state_dict(), os.path.join(args.out_dir, f"{args.model}_fp32.pt"))
    int8_mb = file_size_mb(int8.state_dict(), os.path.join(args.out_dir, f"{args.model}_int8.pt"))

    latency = interleaved_latency({"FP32": fp32, "INT8": int8}, args.rounds, args.repeats,
                                  args.warmup, args.threads, args.seed)
    ratio = paired_round_ratio(latency, "INT8", "FP32")

    result = {
        "experiment": f"{args.model} INT8 post-training static quantization",
        "model": args.model,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "quantization_engine": args.engine,
        "calibration_samples": seen,
        "calibration_source": "validation split, non-augmented",
        "test_samples": len(test_loader.dataset),
        "fp32": {"validation_accuracy_percent": fp32_val, "test_accuracy_percent": fp32_test,
                 "file_size_mb": fp32_mb, "tensor_payload_bytes": fp32_payload_bytes(fp32.state_dict()),
                 "latency": latency["FP32"]},
        "int8": {"validation_accuracy_percent": int8_val, "test_accuracy_percent": int8_test,
                 "file_size_mb": int8_mb, "latency": latency["INT8"]},
        "comparison": {
            "accuracy_change_percentage_points": int8_test - fp32_test,
            "file_size_reduction_percent": 100.0 * (1 - int8_mb / fp32_mb),
            "median_latency_ratio_int8_over_fp32": ratio["median_ratio"],
            "per_round_ratio": ratio,
        },
        "environment": environment_info(),
        "protocol": {"rounds": args.rounds, "repeats_per_round": args.repeats, "warmup": args.warmup,
                     "threads": args.threads, "order": "interleaved, randomised per round"},
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("fp32", "int8", "comparison")}, indent=2, default=str)[:3000])
    print(f"Saved: {args.out_dir}/metrics.json")


if __name__ == "__main__":
    main()
