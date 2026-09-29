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


def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    hard_loss = nn.functional.cross_entropy(student_logits, labels)
    soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
    soft_student = nn.functional.log_softmax(student_logits / temperature, dim=1)
    soft_loss = nn.functional.kl_div(soft_student, soft_targets, reduction="batchmean") * (temperature ** 2)
    return alpha * hard_loss + (1.0 - alpha) * soft_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="results_v2/SmallCNNv2.pt")
    parser.add_argument("--out-dir", default="results_distill")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--total-train", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-subset", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = make_loaders(
        args.data_dir, args.total_train, args.val_size, args.test_subset,
        args.batch_size, args.seed
    )

    teacher = SmallCNNv2().to(device)
    teacher.load_state_dict(torch.load(args.teacher, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = NanoCNNv2().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = -1.0
    best_state = None
    history = {"train_loss": [], "val_accuracy": []}
    start = time.perf_counter()

    for epoch in range(args.epochs):
        student.train()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.inference_mode():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = distillation_loss(
                student_logits, teacher_logits, labels,
                temperature=args.temperature, alpha=args.alpha
            )
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * images.size(0)
            seen += images.size(0)

        scheduler.step()
        train_loss = running / max(seen, 1)
        val_acc = accuracy(student, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        print(f"epoch {epoch + 1:02d}/{args.epochs}: loss={train_loss:.4f} | val_acc={val_acc:.2f}%")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    if best_state is not None:
        student.load_state_dict(best_state)

    train_time = time.perf_counter() - start
    test_acc = accuracy(student, test_loader, device)
    path = os.path.join(args.out_dir, "NanoCNNv2_distilled.pt")
    size_mb = model_size_mb(student, path)
    latency = cpu_latency_ms(student, args.latency_repeats, args.latency_warmup, args.cpu_threads)

    result = {
        "model": "NanoCNNv2_distilled",
        "teacher": "SmallCNNv2",
        "best_validation_accuracy_percent": best_val,
        "test_accuracy_percent": test_acc,
        "trainable_parameters": count_parameters(student),
        "model_size_mb": size_mb,
        "cpu_latency_ms_per_image": latency,
        "training_time_seconds": train_time,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"experiment": "Knowledge distillation", "settings": vars(args), "result": result}, f, indent=2)

    print("\n=== Distillation result ===")
    print(result)


if __name__ == "__main__":
    main()
