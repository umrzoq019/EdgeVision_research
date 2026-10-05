"""Rebuild every table, statistic and figure from the preserved raw results.

    python src/final_analysis.py

Reads  : results/multi_seed, results/distillation, results/quantization, results/v2
Writes : results/analysis/*.csv|json|txt  and  figures/*.png

Design notes (see docs/KNOWN_ISSUES.md):
* Distilled NanoCNNv2 has the same architecture as NanoCNNv2, so latency is
  attributed to the architecture, not measured separately.
* Latency from the legacy sequential benchmark is shown next to the noisy
  end-of-training measurements so the disagreement is visible.
* Paired statistics use per-seed differences (n = number of seeds).
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
except ImportError:  # closed forms for df=2 are used as a fallback
    sps = None

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = {"SmallCNNv2": [32, 64, 96], "TinyCNNv2": [16, 24], "NanoCNNv2": [8, 12]}
ARCH = {"NanoCNNv2_distilled": "NanoCNNv2", "NanoCNNv2_INT8": "NanoCNNv2"}
COLORS = {"SmallCNNv2": "#1f77b4", "TinyCNNv2": "#2ca02c", "NanoCNNv2": "#d62728",
          "NanoCNNv2_distilled": "#ff7f0e", "NanoCNNv2_INT8": "#7f7f7f"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fp32_payload_bytes(model, classes=10):
    """Bytes of parameters + BN buffers (excludes file-container overhead)."""
    c_in, params, buffers = 3, 0, 0
    for c in CHANNELS[model]:
        params += c_in * c * 9 + 2 * c
        buffers += 2 * c
        c_in = c
    params += c_in * classes + classes
    return 4 * (params + buffers) + 8 * len(CHANNELS[model])


def binom_se_pp(p_percent, n):
    p = p_percent / 100.0
    return 100.0 * math.sqrt(p * (1 - p) / n)


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf")
    df = n - 1
    if sps is not None:
        p = float(2 * sps.t.sf(abs(t), df))
        tcrit = float(sps.t.ppf(0.975, df))
    elif df == 2:
        p = 1 - abs(t) / math.sqrt(t * t + 2)
        tcrit = 4.303
    else:
        p, tcrit = float("nan"), float("nan")
    return {"n": n, "differences_pp": d.round(3).tolist(), "mean_diff_pp": mean, "sd_diff_pp": sd,
            "t": t, "df": df, "p_two_sided": p, "ci95_pp": [mean - tcrit * se, mean + tcrit * se]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", default=str(ROOT / "results/multi_seed"))
    ap.add_argument("--distill-dir", default=str(ROOT / "results/distillation/multi_seed"))
    ap.add_argument("--ablation", default=str(ROOT / "results/distillation/ablation_metrics.json"))
    ap.add_argument("--quant", default=str(ROOT / "results/quantization/metrics.json"))
    ap.add_argument("--latency", default=str(ROOT / "results/v2/latency_repeated.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "results/analysis"))
    ap.add_argument("--fig-dir", default=str(ROOT / "figures"))
    ap.add_argument("--test-n", type=int, default=1000, help="test images used (for binomial SE)")
    args = ap.parse_args()
    out, figs = Path(args.out_dir), Path(args.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    # ---------------- per-seed table ----------------
    rows = []
    for seed_result in load(Path(args.baseline_dir) / "raw_results.json"):
        for r in seed_result["results"]:
            rows.append({"type": "baseline", "seed": seed_result["seed"], "model": r["model"],
                         "test_accuracy": r["test_accuracy_percent"], "val_accuracy": r["best_validation_accuracy_percent"],
                         "parameters": r["trainable_parameters"], "size_mb": r["model_size_mb"],
                         "latency_ms": r["cpu_latency_ms_per_image"], "training_time_s": r["training_time_seconds"]})
    for mp in sorted(Path(args.distill_dir).glob("seed_*/metrics.json")):
        d = load(mp)
        r, seed = d["result"], d["settings"]["seed"]
        rows.append({"type": "distilled", "seed": seed, "model": r["model"],
                     "test_accuracy": r["test_accuracy_percent"], "val_accuracy": r["best_validation_accuracy_percent"],
                     "parameters": r["trainable_parameters"], "size_mb": r["model_size_mb"],
                     "latency_ms": r["cpu_latency_ms_per_image"], "training_time_s": r["training_time_seconds"]})
    per_seed = pd.DataFrame(rows).sort_values(["type", "model", "seed"], ascending=[True, True, True])
    per_seed.to_csv(out / "all_multiseed_results.csv", index=False)

    # ---------------- summary + efficiency ----------------
    legacy = load(args.latency)["models"]
    median_latency = {m: legacy[m]["median_ms_per_image"] for m in CHANNELS}  # architecture -> ms (single session)
    summ = []
    for (typ, model), g in per_seed.groupby(["type", "model"], sort=False):
        arch = ARCH.get(model, model)
        summ.append({
            "type": typ, "model": model, "n_seeds": len(g),
            "test_accuracy_mean": g.test_accuracy.mean(), "test_accuracy_sd": g.test_accuracy.std(ddof=1),
            "val_accuracy_mean": g.val_accuracy.mean(), "val_accuracy_sd": g.val_accuracy.std(ddof=1),
            "binomial_se_pp_single_run": binom_se_pp(g.test_accuracy.mean(), args.test_n),
            "parameters": int(g.parameters.iloc[0]), "file_size_mb": g.size_mb.mean(),
            "fp32_payload_bytes": fp32_payload_bytes(arch),
            "latency_end_of_training_mean_ms": g.latency_ms.mean(), "latency_end_of_training_sd_ms": g.latency_ms.std(ddof=1),
            "latency_median_sequential_benchmark_ms": median_latency[arch],
            "latency_source": "architecture median (sequential benchmark, one session)"})
    summary = pd.DataFrame(summ)
    order = {"SmallCNNv2": 0, "TinyCNNv2": 1, "NanoCNNv2": 2, "NanoCNNv2_distilled": 3}
    summary = summary.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)
    summary.to_csv(out / "multiseed_summary.csv", index=False)

    small = summary[summary.model == "SmallCNNv2"].iloc[0]
    eff = summary.copy()
    eff["accuracy_drop_vs_small_pp"] = small.test_accuracy_mean - eff.test_accuracy_mean
    eff["accuracy_retained_percent"] = 100 * eff.test_accuracy_mean / small.test_accuracy_mean
    eff["parameter_reduction_percent"] = 100 * (1 - eff.parameters / small.parameters)
    eff["file_size_reduction_percent"] = 100 * (1 - eff.file_size_mb / small.file_size_mb)
    eff["payload_reduction_percent"] = 100 * (1 - eff.fp32_payload_bytes / small.fp32_payload_bytes)
    eff["speedup_vs_small_sequential_median"] = small.latency_median_sequential_benchmark_ms / eff.latency_median_sequential_benchmark_ms
    eff["accuracy_per_1000_params"] = eff.test_accuracy_mean / (eff.parameters / 1000)
    eff.to_csv(out / "efficiency_analysis.csv", index=False)

    # ---------------- paired statistics ----------------
    def vec(model):
        return per_seed[per_seed.model == model].sort_values("seed").test_accuracy.values

    comparisons = {"NanoCNNv2_distilled - NanoCNNv2": ("NanoCNNv2_distilled", "NanoCNNv2"),
                   "TinyCNNv2 - NanoCNNv2": ("TinyCNNv2", "NanoCNNv2"),
                   "SmallCNNv2 - TinyCNNv2": ("SmallCNNv2", "TinyCNNv2"),
                   "SmallCNNv2 - NanoCNNv2": ("SmallCNNv2", "NanoCNNv2")}
    stats = {k: paired(vec(a), vec(b)) for k, (a, b) in comparisons.items()}
    (out / "paired_statistics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    # ---------------- Pareto (accuracy up, cost down) by mean ----------------
    def non_dominated(df, cost):
        flags = []
        for i, r in df.iterrows():
            dom = ((df.test_accuracy_mean >= r.test_accuracy_mean) & (df[cost] <= r[cost]) &
                   ((df.test_accuracy_mean > r.test_accuracy_mean) | (df[cost] < r[cost]))).any()
            flags.append(not dom)
        return flags
    par = summary[["type", "model", "test_accuracy_mean", "test_accuracy_sd", "parameters", "fp32_payload_bytes"]].copy()
    par["non_dominated_acc_vs_params"] = non_dominated(par, "parameters")
    par["non_dominated_acc_vs_payload"] = non_dominated(par, "fp32_payload_bytes")
    par["note"] = "by means only; NanoCNNv2_distilled vs NanoCNNv2 is not statistically distinguishable"
    par.to_csv(out / "pareto_analysis.csv", index=False)

    # ---------------- distillation ablation ----------------
    ab = pd.DataFrame(load(args.ablation)["results"])
    ab["rank_by_validation"] = ab.best_validation_accuracy_percent.rank(ascending=False, method="min").astype(int)
    ab["rank_by_test"] = ab.test_accuracy_percent.rank(ascending=False, method="min").astype(int)
    ref = float(per_seed[(per_seed.model == "NanoCNNv2") & (per_seed.seed == 42)].test_accuracy.iloc[0])
    ab["delta_vs_seed42_baseline_pp"] = ab.test_accuracy_percent - ref
    ab = ab.rename(columns={"best_validation_accuracy_percent": "best_validation_accuracy",
                            "test_accuracy_percent": "test_accuracy", "cpu_latency_ms_per_image": "latency_ms_unreliable",
                            "model_size_mb": "size_mb", "training_time_seconds": "training_time_s", "trainable_parameters": "parameters"})
    ab.to_csv(out / "distillation_ablation.csv", index=False)
    same_cfg = float(per_seed[(per_seed.type == "distilled") & (per_seed.seed == 42)].test_accuracy.iloc[0])
    t2s50 = float(ab[ab.method == "T2_soft50"].test_accuracy.iloc[0])

    # ---------------- quantization ----------------
    q = load(args.quant)
    fr, ir = q["fp32"]["latency"]["rounds"], q["int8"]["latency"]["rounds"]
    qsum = {"model": q["model"], "seed": q["seed"], "n_seeds": 1,
            "fp32_test_accuracy_percent": q["fp32"]["test_accuracy_percent"],
            "int8_test_accuracy_percent": q["int8"]["test_accuracy_percent"],
            "accuracy_change_pp": q["int8"]["test_accuracy_percent"] - q["fp32"]["test_accuracy_percent"],
            "binomial_se_pp_single_run": binom_se_pp(q["fp32"]["test_accuracy_percent"], args.test_n),
            "fp32_file_mb": q["fp32"]["size_mb"], "int8_file_mb": q["int8"]["size_mb"],
            "file_size_reduction_percent": 100 * (1 - q["int8"]["size_mb"] / q["fp32"]["size_mb"]),
            "fp32_payload_bytes_expected": fp32_payload_bytes("NanoCNNv2"),
            "fp32_median_ms": q["fp32"]["latency"]["median_ms_per_image"],
            "int8_median_ms": q["int8"]["latency"]["median_ms_per_image"],
            "median_ratio_int8_over_fp32": q["int8"]["latency"]["median_ms_per_image"] / q["fp32"]["latency"]["median_ms_per_image"],
            "fp32_rounds_max_over_min": max(fr) / min(fr),
            "per_round_ratio_int8_over_fp32": [round(b / a, 3) for a, b in zip(fr, ir)],
            "latency_conclusion": "inconclusive: FP32 rounds drifted by %.0f%% within the run and FP32/INT8 appear to have been timed one after the other (not interleaved)" % (100 * (max(fr) / min(fr) - 1))}
    (out / "quantization_summary.json").write_text(json.dumps(qsum, indent=2) + "\n", encoding="utf-8")

    # ---------------- final comparison table ----------------
    comp = summary[["type", "model", "test_accuracy_mean", "test_accuracy_sd", "parameters", "file_size_mb", "latency_median_sequential_benchmark_ms"]].copy()
    comp.columns = ["category", "model", "test_accuracy_percent", "accuracy_sd", "parameters", "file_size_mb", "latency_ms_architecture_median"]
    comp = pd.concat([comp, pd.DataFrame([{"category": "quantized (seed 42 only)", "model": "NanoCNNv2_INT8",
            "test_accuracy_percent": qsum["int8_test_accuracy_percent"], "accuracy_sd": np.nan, "parameters": 1250,
            "file_size_mb": qsum["int8_file_mb"], "latency_ms_architecture_median": np.nan}])], ignore_index=True)
    comp.to_csv(out / "final_experiment_comparison.csv", index=False)

    # ---------------- text report ----------------
    pd.set_option("display.width", 220, "display.max_columns", 40)
    lines = ["EdgeVision final analysis (generated by src/final_analysis.py)", "=" * 78, "",
             "MULTI-SEED SUMMARY (3 seeds; test = first 1,000 CIFAR-10 test images)", "-" * 78,
             summary[["model", "n_seeds", "test_accuracy_mean", "test_accuracy_sd", "val_accuracy_mean", "parameters",
                      "file_size_mb", "fp32_payload_bytes"]].round(4).to_string(index=False), "",
             "EFFICIENCY vs SmallCNNv2", "-" * 78,
             eff[["model", "accuracy_drop_vs_small_pp", "accuracy_retained_percent", "parameter_reduction_percent",
                  "payload_reduction_percent", "speedup_vs_small_sequential_median"]].round(3).to_string(index=False), "",
             "PAIRED STATISTICS (per-seed differences in test accuracy, pp)", "-" * 78]
    for k, v in stats.items():
        lines.append(f"{k}: diffs={v['differences_pp']} mean={v['mean_diff_pp']:+.2f} sd={v['sd_diff_pp']:.2f} "
                     f"t={v['t']:.2f} (df={v['df']}) p={v['p_two_sided']:.3f} 95% CI=[{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}]")
    lines += ["", "DISTILLATION ABLATION (single seed 42; independent NanoCNNv2 seed-42 baseline = %.1f%%)" % ref, "-" * 78,
              ab[["method", "temperature", "soft_weight", "best_validation_accuracy", "test_accuracy",
                  "rank_by_validation", "rank_by_test"]].to_string(index=False),
              f"Same nominal config (T=2, soft 0.5, seed 42): {t2s50:.1f}% in the ablation vs {same_cfg:.1f}% in the multi-seed run "
              f"-> run-to-run gap {abs(t2s50 - same_cfg):.1f} pp, comparable to the whole spread across ablation settings "
              f"({ab.test_accuracy.max() - ab.test_accuracy.min():.1f} pp).", "",
              "QUANTIZATION (NanoCNNv2, seed 42 only)", "-" * 78, json.dumps(qsum, indent=2), ""]
    (out / "final_report.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---------------- figures ----------------
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    mk = {"SmallCNNv2": "o", "TinyCNNv2": "s", "NanoCNNv2": "^", "NanoCNNv2_distilled": "D"}

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, r in summary.iterrows():
        ax.errorbar(r.parameters, r.test_accuracy_mean, yerr=r.test_accuracy_sd, fmt=mk[r.model], ms=8, capsize=4,
                    color=COLORS[r.model], label=f"{r.model} (n={r.n_seeds})")
    ax.errorbar(1250, qsum["int8_test_accuracy_percent"], fmt="X", ms=9, color=COLORS["NanoCNNv2_INT8"], label="NanoCNNv2 INT8 (seed 42 only)")
    ax.set_xscale("log"); ax.set_xlabel("Trainable parameters (log scale)"); ax.set_ylabel("Test accuracy (%) - mean ± SD over seeds")
    ax.set_title("Accuracy vs parameters"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figs / "accuracy_vs_parameters.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in CHANNELS:
        r = summary[summary.model == m].iloc[0]
        ax.errorbar(r.latency_end_of_training_mean_ms, r.test_accuracy_mean, xerr=r.latency_end_of_training_sd_ms, yerr=r.test_accuracy_sd,
                    fmt=mk[m], mfc="none", ms=9, capsize=3, color=COLORS[m], alpha=0.8,
                    label=f"{m}: end-of-training timing, mean ± SD" if m == "SmallCNNv2" else None)
        ax.plot(r.latency_median_sequential_benchmark_ms, r.test_accuracy_mean, mk[m], ms=9, color=COLORS[m], label=f"{m}: sequential benchmark median")
        ax.annotate(m, (r.latency_median_sequential_benchmark_ms, r.test_accuracy_mean), xytext=(7, 7), textcoords="offset points", fontsize=8)
    ax.set_xlabel("CPU latency, 1 thread (ms/image)"); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs latency (hollow = noisy end-of-training timing)"); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(figs / "accuracy_vs_latency.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in CHANNELS:
        r = summary[summary.model == m].iloc[0]
        ax.plot(r.file_size_mb * 1024, r.test_accuracy_mean, mk[m], ms=9, color=COLORS[m], label=f"{m} file")
        ax.plot(r.fp32_payload_bytes / 1024, r.test_accuracy_mean, mk[m], ms=9, mfc="none", color=COLORS[m], label=f"{m} tensor payload")
    ax.set_xscale("log"); ax.set_xlabel("Size (KiB, log scale)"); ax.set_ylabel("Mean test accuracy (%)")
    ax.set_title("Accuracy vs size (filled = saved file, hollow = parameter bytes)"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(figs / "accuracy_vs_size.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    models = ["SmallCNNv2", "TinyCNNv2", "NanoCNNv2", "NanoCNNv2_distilled"]
    for x, m in enumerate(models):
        v = per_seed[per_seed.model == m].sort_values("seed")
        ax.scatter([x] * len(v), v.test_accuracy, s=45, color=COLORS[m], zorder=3)
        ax.hlines(v.test_accuracy.mean(), x - 0.25, x + 0.25, color=COLORS[m], lw=2)
    for k, seed in enumerate(sorted(per_seed.seed.unique())):
        a = per_seed[(per_seed.model == "NanoCNNv2") & (per_seed.seed == seed)].test_accuracy.iloc[0]
        b = per_seed[(per_seed.model == "NanoCNNv2_distilled") & (per_seed.seed == seed)].test_accuracy.iloc[0]
        ax.plot([2, 3], [a, b], color="k", alpha=0.35, lw=1); ax.annotate(f"seed {seed}", (3.05, b), xytext=(2, (-6, 0, 6)[k % 3]), textcoords="offset points", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Small", "Tiny", "Nano", "Nano + KD"]); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Per-seed results (lines pair Nano and Nano+KD for the same seed)"); fig.tight_layout()
    fig.savefig(figs / "seed_variability.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(ab)); w = 0.38
    ax.bar(x - w / 2, ab.best_validation_accuracy, w, label="validation (used for checkpoint choice)", color="#9ecae1")
    ax.bar(x + w / 2, ab.test_accuracy, w, label="test (1,000 images)", color="#3182bd")
    ax.axhline(ref, color="k", ls="--", lw=1, label=f"independent Nano, seed 42 (test {ref:.1f}%)")
    ax.axhspan(ref - binom_se_pp(ref, args.test_n), ref + binom_se_pp(ref, args.test_n), color="k", alpha=0.08, label="±1 binomial SE (test)")
    ax.set_xticks(x); ax.set_xticklabels(ab.method); ax.set_ylim(30, 40); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Distillation ablation (single seed): differences are within noise"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(figs / "distillation_ablation.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(range(1, len(fr) + 1), fr, "o-", label="FP32", color=COLORS["NanoCNNv2"])
    ax.plot(range(1, len(ir) + 1), ir, "s-", label="INT8", color=COLORS["NanoCNNv2_INT8"])
    ax.set_xlabel("Round (FP32 and INT8 apparently timed in separate blocks)"); ax.set_ylabel("ms/image")
    ax.set_title("NanoCNNv2 FP32 vs INT8 latency by round: drift makes the comparison inconclusive"); ax.legend()
    ax.title.set_fontsize(9); fig.tight_layout(); fig.savefig(figs / "quantization_latency_rounds.png", dpi=170); plt.close(fig)

    print("\n".join(lines[:40]))
    print(f"\nWrote tables to {out} and figures to {figs}")


if __name__ == "__main__":
    main()
