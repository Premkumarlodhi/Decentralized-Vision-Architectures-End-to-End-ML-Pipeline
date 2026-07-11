# Decentralized Vision Architectures — End-to-End ML Pipeline

> **Summer of Code (SoC) Project**
> Mentored project under the **Analytics Club, IIT Bombay**
> [GitHub Repository](https://github.com/Premkumarlodhi/Decentralized-Vision-Architectures-End-to-End-ML-Pipeline)

---

A production-structured Federated Learning pipeline built on PyTorch and Flower, studying how aggregation strategy choice affects model quality under realistic Non-IID agricultural data distributions. The project simulates a decentralized vision system where 10 independent clients (representing geographically distributed farms) collaboratively train a plant disease classifier without sharing raw data.

The core contribution is a rigorous benchmarking study: FedAvg vs. FedProx across two heterogeneity levels, with Top-k gradient sparsification measuring the accuracy-communication tradeoff throughout.

---

## Results at a Glance

| Strategy  | α = 0.1 (High Skew) | α = 1.0 (Low Skew) | FedProx Gain (α=0.1) |
|-----------|--------------------|--------------------|----------------------|
| FedAvg    | 51.3%              | 74.1%              | —                    |
| FedProx   | 63.4%              | 76.2%              | **+12.1 pp**         |

**Communication:** Top-30% gradient sparsification reduced per-experiment payload from ~9.74 MB to ~2.92 MB (**70% reduction**) with under 2 pp accuracy degradation across all strategies and heterogeneity levels.

---

## What This Project Studies

Standard FL benchmarks use IID data — every client sees a balanced, representative sample of all classes. Real agricultural deployments don't work that way. A farm in one region grows one crop and sees a handful of diseases; a farm elsewhere has an entirely different distribution.

This project simulates that reality using Dirichlet-distributed data partitioning (controlled by concentration parameter α) and answers two questions:

- How much does FedAvg degrade under extreme client data heterogeneity, and does FedProx meaningfully close that gap?
- At what compression ratio does Top-k gradient sparsification begin to hurt global accuracy, and is that threshold consistent across strategies?

---

## Key Findings

**FedProx's advantage is heterogeneity-dependent.** At α=0.1, FedProx outperforms FedAvg by 12.1 percentage points. At α=1.0, the gap shrinks to 2.1 pp — well within the noise of 50% partial client participation. The proximal penalty only earns its convergence cost under genuine data skew.

**FedAvg stagnates mid-training under high skew.** At α=0.1, FedAvg's accuracy plateaus around round 6 (~47%) before recovering marginally. FedProx maintains a steeper climb through round 9, suggesting the proximal term actively prevents client weight drift during the critical mid-training phase.

**70% communication reduction costs under 2 pp accuracy.** Transmitting only the classifier head (48K parameters vs. 2.2M frozen backbone parameters) combined with Top-30% sparsification keeps per-experiment payloads under 3 MB with negligible model quality loss.

---

## Architecture

### Why MobileNetV2 with a Frozen Backbone

Only the classification head (48,678 parameters) is federated — the ImageNet-pretrained backbone is frozen on all clients. This reflects how FL is deployed on constrained edge devices: local adaptation happens at the task-specific layer, not the full feature extractor. It also isolates the communication cost to a small, well-defined tensor, making compression results cleaner to interpret.

### Why Dirichlet Partitioning

The concentration parameter α controls how skewed each client's class distribution is. At α → ∞ you recover IID; at α = 0.1 some clients hold 65–70% of their samples from a single disease class. A single parameter reproduces both the tutorial IID case and the realistic agricultural case.

### Why Partial Participation

Each round, 5 of 10 clients are selected at random. This forces the server's aggregation logic to handle incomplete participation every round — a system that only works when all clients respond is not production-ready.

---

## Repository Structure

```
federated-plantvillage/
├── config.py                  # All hyperparameters — single source of truth
├── data/
│   └── partitioning.py        # Dirichlet splitter across 10 clients
├── models/
│   └── mobilenet.py           # MobileNetV2, frozen backbone, federated head
├── client/
│   └── worker.py              # Local training + FedProx term + Top-k compression
├── server/
│   └── strategy.py            # FedAvg and FedProx strategies (shared aggregation)
├── utils/
│   └── metrics.py             # CSV logging + comparison plot generation
├── cache/                     # Auto-generated partition indices (gitignored)
├── results/
│   ├── metrics.csv            # 40 rows — 4 experiments × 10 rounds
│   └── comparison.png         # Accuracy curves + communication cost figure
├── main.py                    # CLI entry point
├── benchmark_report.md        # Full experimental analysis
└── README.md
```

---

## Setup

### Prerequisites

```bash
python >= 3.9
torch >= 2.0
torchvision >= 0.15
flwr >= 1.5
```

### Installation

```bash
git clone https://github.com/Premkumarlodhi/Decentralized-Vision-Architectures-End-to-End-ML-Pipeline.git
cd Decentralized-Vision-Architectures-End-to-End-ML-Pipeline
pip install -r requirements.txt
```

### Dataset

Download [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) and place it at `PlantVillage/` in the project root. The directory should contain one subfolder per disease class — `datasets.ImageFolder` handles the rest automatically.

```
federated-plantvillage/
└── PlantVillage/
    ├── Pepper__bell__Bacterial_spot/
    ├── Potato__Early_blight/
    ├── Tomato__healthy/
    └── ...
```

---

## Running Experiments

Run all four experiments in sequence. Each appends one row per round to `results/metrics.csv`. Partition indices are cached to disk after the first run at each α value — subsequent runs at the same α skip recomputation.

```bash
# High heterogeneity (α=0.1)
python main.py --strategy fedavg  --alpha 0.1
python main.py --strategy fedprox --alpha 0.1

# Low heterogeneity (α=1.0)
python main.py --strategy fedavg  --alpha 1.0
python main.py --strategy fedprox --alpha 1.0
```

Generate the comparison figure after all four runs:

```bash
python -c "from utils.metrics import plot; plot()"
```

Output saved to `results/comparison.png`.

### Configuration

All hyperparameters live in `config.py`. Change experiments by editing one file:

```python
ALPHA             = 0.1    # Dirichlet concentration — lower = more skew
NUM_CLIENTS       = 10
CLIENTS_PER_ROUND = 5      # Partial participation — 50% each round
ROUNDS            = 10
LOCAL_EPOCHS      = 1
MU                = 0.01   # FedProx proximal coefficient
TOP_K             = 0.3    # Transmit top 30% of gradient deltas
```

---

## Implementation Notes

### FedProx Proximal Term

Applied client-side in `client/worker.py`. At each local step:

```
loss = CrossEntropy(output, label) + (μ/2) × Σ||w - w_global||²
```

`w_global` is the set of global weights received from the server at the start of the round, frozen during local training and used only for the penalty computation. Server-side aggregation is identical to FedAvg — the only algorithmic difference between the two strategies is this client-side term.

### Top-k Gradient Sparsification

After local training, compute `delta = w_new - w_global`. Flatten the delta across all parameters into one vector, keep the top 30% by absolute magnitude, zero out the rest. Only the sparse delta is transmitted. The server applies the delta to the global model before the next round.

### Caching Strategy

Dirichlet partitioning is computed once per α value and saved as `.npy` index files under `cache/`. Workers load only their own client's indices from disk — this avoids re-running the full partition computation inside Ray actor processes on every round, which caused significant slowdowns on Windows due to Ray's `spawn`-based process model.

---

## Benchmark Report

Full experimental analysis, per-round accuracy tables, and interpretation of results:
[`benchmark_report.md`](./benchmark_report.md)

---

## Limitations and Extensions

**10 rounds is short.** Both strategies show upward trends at round 10 with no clear plateau — extended training (25–50 rounds) would likely widen the FedProx advantage under high skew.

**μ was not ablated.** Sensitivity analysis across μ ∈ {0.001, 0.01, 0.1} is a natural next step — the optimal proximal coefficient likely varies with α.

**Frozen backbone caps accuracy.** The gap between these federated results (51–76%) and a fully fine-tuned centralised MobileNetV2 on PlantVillage (>95%) is largely attributable to the frozen backbone constraint, not the FL setup.

**Natural extensions:**
- Ablate μ across heterogeneity levels to find the μ-α interaction
- Add SCAFFOLD as a third strategy for a complete comparison
- Federate the backbone (unfreeze) and measure the communication cost increase vs. accuracy gain
- Extend to 25 rounds to observe convergence plateaus

---

## Project Context

This project was developed as part of **Summer of Code (SoC)**, a mentorship programme by the **Analytics Club, IIT Bombay**, where students independently build and benchmark ML systems over the summer under mentor guidance.

The project began from a basic sequential Flower notebook (IID split, 3 clients, vanilla FedAvg) and was restructured into a modular, benchmarking-oriented pipeline studying the practical tradeoffs of federated aggregation strategies under realistic data constraints.

---

## References

- McMahan et al. (2017) — [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) — FedAvg
- Li et al. (2020) — [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) — FedProx
- Beutel et al. (2020) — [Flower: A Friendly Federated Learning Framework](https://arxiv.org/abs/2007.14390)
- Hughes & Salathé (2015) — [An open access repository of images on plant health](https://arxiv.org/abs/1511.08060) — PlantVillage

---

## Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Flower](https://img.shields.io/badge/Flower-FL-brightgreen?style=flat)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Ray](https://img.shields.io/badge/Ray-028CF0?style=flat&logo=ray&logoColor=white)
