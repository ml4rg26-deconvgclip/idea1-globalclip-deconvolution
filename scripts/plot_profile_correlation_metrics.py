#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

run_dir = Path("results/reconstruction_head/interphase_50")
csv_path = run_dir / "profile_correlation_metrics.csv"
out_dir = run_dir / "report_assets"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

# 只画 test set
df = df[df["split"] == "test"].copy()

# 模型顺序
model_order = [
    "reconstruction_head",
    "parnet_unweighted_average",
    "uniform_valid_profile",
    "train_mean_observed_profile",
]
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values("model")

# 更好看的标签
label_map = {
    "reconstruction_head": "Reconstruction head",
    "parnet_unweighted_average": "Unweighted Parnet average",
    "uniform_valid_profile": "Uniform valid profile",
    "train_mean_observed_profile": "Train mean observed profile",
}
df["label"] = df["model"].map(label_map)

# 要画的指标
plots = [
    ("mean_pearson", "Mean Pearson correlation", "profile_mean_pearson_test.png"),
    ("mean_spearman", "Mean Spearman correlation", "profile_mean_spearman_test.png"),
    ("mean_mse", "Mean MSE", "profile_mean_mse_test.png"),
]

for metric, ylabel, filename in plots:
    plt.figure(figsize=(8, 5))
    plt.bar(df["label"], df[metric])
    plt.ylabel(ylabel)
    plt.title(f"Test set comparison: {ylabel}")
    plt.xticks(rotation=20, ha="right")

    # 标数值
    for i, v in enumerate(df[metric]):
        if pd.notna(v):
            plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=300)
    plt.savefig(out_dir / filename.replace(".png", ".pdf"))
    plt.close()

print(f"Saved plots to: {out_dir}")
