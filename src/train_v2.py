import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 42
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class SmallCNNv2(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class TinyCNNv2(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(24, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class NanoCNNv2(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 12, 3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(12, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


MODELS = {
    "SmallCNNv2": SmallCNNv2,
    "TinyCNNv2": TinyCNNv2,
    "NanoCNNv2": NanoCNNv2,
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(data_dir: str, total_train: int, val_size: int, test_subset: int,
                 batch_size: int, seed: int):
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    augmented = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tfms)
    clean = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_tfms)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tfms)

    total_train = min(total_train, len(augmented))
    val_size = min(val_size, total_train - 1)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_train, generator=generator).tolist()
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_ds = Subset(augmented, train_idx)
    val_ds = Subset(clean, val_idx)
    test_ds = Subset(test, list(range(min(test_subset, len(test)))))

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += int((preds == labels).sum())
            total += labels.numel()
    return 100.0 * correct / max(total, 1)


def train_model(model, train_loader, val_loader, device, epochs, lr, weight_decay):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = -1.0
    best_state = None
    history = {"train_loss": [], "val_accuracy": [], "learning_rate": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)

        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        val_acc = accuracy(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        history["learning_rate"].append(current_lr)

        print(f"epoch {epoch + 1:02d}/{epochs}: loss={train_loss:.4f} | val_acc={val_acc:.2f}% | lr={current_lr:.6f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val


def model_size_mb(model: nn.Module, path: str) -> float:
    torch.save(model.state_dict(), path)
    return os.path.getsize(path) / (1024 * 1024)


def cpu_latency_ms(model: nn.Module, repeats: int = 100, warmup: int = 20, threads: int = 1) -> float:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        elapsed = time.perf_counter() - start
    torch.set_num_threads(previous_threads)
    return elapsed * 1000.0 / repeats


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_one(name, model, train_loader, val_loader, test_loader, device, args, out_dir):
    print(f"\n=== {name} ===")
    start = time.perf_counter()
    history, best_val = train_model(
        model, train_loader, val_loader, device,
        args.epochs, args.lr, args.weight_decay
    )
    train_time = time.perf_counter() - start
    test_acc = accuracy(model, test_loader, device)

    ckpt_path = os.path.join(out_dir, f"{name}.pt")
    size_mb = model_size_mb(model, ckpt_path)
    latency = cpu_latency_ms(model, args.latency_repeats, args.latency_warmup, args.cpu_threads)
    params = count_parameters(model)

    result = {
        "model": name,
        "best_validation_accuracy_percent": best_val,
        "test_accuracy_percent": test_acc,
        "trainable_parameters": params,
        "model_size_mb": size_mb,
        "cpu_latency_ms_per_image": latency,
        "training_time_seconds": train_time,
        "history": history,
    }
    print(
        f"test_acc={test_acc:.2f}% | params={params:,} | size={size_mb:.3f} MB | "
        f"CPU={latency:.3f} ms/image"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="SmallCNNv2,TinyCNNv2,NanoCNNv2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--total-train", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-subset", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results_v2")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = make_loaders(
        args.data_dir, args.total_train, args.val_size, args.test_subset,
        args.batch_size, args.seed
    )

    results = []
    for model_name in [x.strip() for x in args.models.split(",") if x.strip()]:
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        set_seed(args.seed)
        model = MODELS[model_name]().to(device)
        results.append(run_one(model_name, model, train_loader, val_loader, test_loader, device, args, args.out_dir))

    payload = {
        "experiment": "EdgeVision v2 - stronger training",
        "device": str(device),
        "seed": args.seed,
        "settings": vars(args),
        "results": results,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.out_dir}/metrics.json")


if __name__ == "__main__":
    main()
