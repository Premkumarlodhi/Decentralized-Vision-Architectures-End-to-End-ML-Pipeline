# main.py

import os
import argparse
import numpy as np
import torch
import flwr as fl
from torchvision import datasets, transforms
from torch.utils.data import Subset

from config            import (NUM_CLIENTS, CLIENTS_PER_ROUND, ROUNDS,
                                MU, TOP_K, DATA_DIR)
from data.partitioning import get_client_datasets
from models.mobilenet  import get_model
from client.worker     import train, evaluate, compress_delta
from server.strategy   import FedAvgStrategy, FedProxStrategy
from utils.metrics     import log_round

# ── args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", choices=["fedavg", "fedprox"], required=True)
parser.add_argument("--alpha",    type=float, required=True)
args = parser.parse_args()

# ── device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── partition once, cache indices to disk ─────────────────────────────────────
cache_tag = f"cache/alpha_{args.alpha}"

if not os.path.exists(f"{cache_tag}_train_0.npy"):
    print(f"Partitioning dataset at alpha={args.alpha}...")
    all_datasets, NUM_CLASSES = get_client_datasets(alpha=args.alpha)
    os.makedirs("cache", exist_ok=True)
    for i, (tr, te) in enumerate(all_datasets):
        np.save(f"{cache_tag}_train_{i}.npy", np.array(tr.indices))
        np.save(f"{cache_tag}_test_{i}.npy",  np.array(te.indices))

    # save num_classes so we can reload without re-partitioning
    np.save(f"{cache_tag}_meta.npy", np.array([NUM_CLASSES]))
    print(f"Cached. num_classes={NUM_CLASSES}")
else:
    NUM_CLASSES = int(np.load(f"{cache_tag}_meta.npy")[0])
    print(f"Loaded cached partition. num_classes={NUM_CLASSES}")

GLOBAL_MODEL = get_model(NUM_CLASSES).to(DEVICE)

# ── client factory ────────────────────────────────────────────────────────────
def client_fn(context):
    cid   = int(context.node_config["partition-id"])
    alpha = args.alpha

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    full_dataset  = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_indices = np.load(f"cache/alpha_{alpha}_train_{cid}.npy")
    test_indices  = np.load(f"cache/alpha_{alpha}_test_{cid}.npy")
    train_set     = Subset(full_dataset, train_indices.tolist())
    test_set      = Subset(full_dataset, test_indices.tolist())

    model = get_model(NUM_CLASSES).to(DEVICE)

    class _Client(fl.client.NumPyClient):
        def get_parameters(self, config=None):
            return [
                p.detach().cpu().numpy()
                for p in model.parameters()
                if p.requires_grad
            ]

        def set_parameters(self, params):
            trainable = [p for p in model.parameters() if p.requires_grad]
            for p, v in zip(trainable, params):
                p.data = torch.tensor(v).to(DEVICE)

        def fit(self, parameters, config):
            global_weights = [torch.tensor(p) for p in parameters]
            self.set_parameters(parameters)

            mu = MU if args.strategy == "fedprox" else 0.0
            train(model, train_set, DEVICE,
                  global_weights=global_weights, mu=mu)

            new_weights = [
                p.detach().cpu()
                for p in model.parameters()
                if p.requires_grad
            ]
            sparse_deltas, bytes_sent = compress_delta(
                new_weights, global_weights, top_k=TOP_K
            )
            updated = [g + d for g, d in zip(global_weights, sparse_deltas)]
            return (
                [u.numpy() for u in updated],
                len(train_set),
                {"bytes_sent": bytes_sent}
            )

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            acc = evaluate(model, test_set, DEVICE)
            return 1.0 - acc, len(test_set), {"accuracy": acc}

    return _Client().to_client()

# ── strategy ──────────────────────────────────────────────────────────────────
def on_round_end(round_num, accuracy, bytes_sent):
    log_round(args.strategy, args.alpha, round_num, accuracy, bytes_sent)

StrategyClass = FedAvgStrategy if args.strategy == "fedavg" else FedProxStrategy
strategy      = StrategyClass(model_keys=list(GLOBAL_MODEL.state_dict().keys()), on_round_end=on_round_end)

# ── run ───────────────────────────────────────────────────────────────────────
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=ROUNDS),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0.0},
    ray_init_args={"include_dashboard": False, "ignore_reinit_error": True}
)