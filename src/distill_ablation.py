import argparse
import json
import os
import time

import torch
from torch import nn

from train_v2 import (
    SEED, NanoCNNv2, SmallCNNv2, accuracy, cpu_latency_ms,
    count_parameters, make_loaders, model_size_mb, set_seed
)


def kd_loss(student_logits, teacher_logits, labels, temperature, soft_weight):
    ce_weight = 1.0 - soft_weight
    ce = nn.functional.cross_entropy(student_logits, labels)
    teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = nn.functional.log_softmax(student_logits / temperature, dim=1)
    kd = nn.functional.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    ) * (temperature ** 2)
    return ce_weight * ce + soft_weight * kd


def run_one(args, T, soft_weight, tag, teacher, loaders, device):
    train_loader, val_loader, test_loader = loaders
    student = NanoCNNv2().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val = -1.0
    best_state = None
    start = time.perf_counter()

    for epoch in range(args.epochs):
        student.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.inference_mode():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = kd_loss(student_logits, teacher_logits, labels, T, soft_weight)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_acc = accuracy(student, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        print(f"{tag} | epoch {epoch+1:02d}/{args.epochs} | val_acc={val_acc:.2f}%")

    student.load_state_dict(best_state)
    test_acc = accuracy(student, test_loader, device)
    path = os.path.join(args.out_dir, f"NanoCNNv2_distilled_{tag}.pt")
    size_mb = model_size_mb(student, path)
    latency = cpu_latency_ms(student, args.latency_repeats, args.latency_warmup, args.cpu_threads)
    result = {
        "method": tag,
        "temperature": T,
        "soft_weight": soft_weight,
        "cross_entropy_weight": 1.0 - soft_weight,
        "best_validation_accuracy_percent": best_val,
        "test_accuracy_percent": test_acc,
        "trainable_parameters": count_parameters(student),
        "model_size_mb": size_mb,
        "cpu_latency_ms_per_image": latency,
        "training_time_seconds": time.perf_counter() - start,
    }
    print(f"RESULT {tag}: test_acc={test_acc:.2f}% | size={size_mb:.3f} MB | latency={latency:.3f} ms")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="results_v2/SmallCNNv2.pt")
    p.add_argument("--out-dir", default="results_distill_ablation")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--total-train", type=int, default=10000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--test-subset", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--latency-repeats", type=int, default=100)
    p.add_argument("--latency-warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    loaders = make_loaders(args.data_dir, args.total_train, args.val_size, args.test_subset, args.batch_size, args.seed)
    teacher = SmallCNNv2().to(device)
    teacher.load_state_dict(torch.load(args.teacher, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # First setting follows the weighting pattern used in the official PyTorch KD tutorial:
    # 25% soft-target loss and 75% hard-label cross entropy, with T=2.
    settings = [
        (2.0, 0.25, "T2_soft25"),
        (4.0, 0.25, "T4_soft25"),
        (2.0, 0.50, "T2_soft50"),
    ]
    results = []
    for T, soft_weight, tag in settings:
        set_seed(args.seed)
        results.append(run_one(args, T, soft_weight, tag, teacher, loaders, device))

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"experiment": "Knowledge distillation ablation", "results": results}, f, indent=2)
    print(f"Saved: {args.out_dir}/metrics.json")

if __name__ == "__main__":
    main()
