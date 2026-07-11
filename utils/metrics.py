# utils/metrics.py

import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_PATH


def log_round(strategy_name, alpha, round_num, accuracy, bytes_sent):
    os.makedirs("results", exist_ok=True)
    file_exists = os.path.exists(RESULTS_PATH)

    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["strategy", "alpha", "round", "accuracy", "bytes_sent"]
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "strategy":   strategy_name,
            "alpha":      alpha,
            "round":      round_num,
            "accuracy":   round(accuracy, 4),
            "bytes_sent": bytes_sent,
        })


def plot():
    df = pd.read_csv(RESULTS_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FedAvg vs FedProx — PlantVillage", fontsize=13)

    colors = {
        ("fedavg",  0.1): "#4C72B0",
        ("fedprox", 0.1): "#DD8452",
        ("fedavg",  1.0): "#55A868",
        ("fedprox", 1.0): "#C44E52",
    }

    for (strategy, alpha), group in df.groupby(["strategy", "alpha"]):
        axes[0].plot(
            group["round"],
            group["accuracy"],
            label=f"{strategy} α={alpha}",
            marker="o",
            markersize=4,
            color=colors.get((strategy, float(alpha)), None)
        )

    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Global Accuracy")
    axes[0].set_title("Accuracy vs. Communication Rounds")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    summary = (
        df.groupby(["strategy", "alpha"])["bytes_sent"]
        .sum()
        .reset_index()
    )
    labels = [f"{r.strategy}\nα={r.alpha}" for _, r in summary.iterrows()]
    bar_colors = [
        colors.get((r.strategy, float(r.alpha)), "#999999")
        for _, r in summary.iterrows()
    ]
    axes[1].bar(labels, summary["bytes_sent"] / 1e6, color=bar_colors)
    axes[1].set_ylabel("Total MB Transmitted")
    axes[1].set_title("Cumulative Communication Cost")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to results/comparison.png")