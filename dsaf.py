# verify_phase4.py

import os
import numpy as np
import torch
import flwr as fl
from torchvision import datasets, transforms
from torch.utils.data import Subset

from models.mobilenet  import get_model
from data.partitioning import get_client_datasets
from client.worker     import train, evaluate, compress_delta
from server.strategy   import FedAvgStrategy
from config            import TOP_K, DATA_DIR

DEVICE = torch.device("cpu")

# ── Step 1: compute partition indices ONCE and cache to disk ────────────────
print("Partitioning dataset...")
_all_datasets, NUM_CLASSES = get_client_datasets(alpha=0.1)

os.makedirs("cache", exist_ok=True)
for i, (train_set, test_set) in enumerate(_all_datasets):
    np.save(f"cache/train_{i}.npy", np.array(train_set.indices))
    np.save(f"cache/test_{i}.npy",  np.array(test_set.indices))

print(f"Cached indices for {len(_all_datasets)} clients. num_classes={NUM_CLASSES}")

GLOBAL_MODEL = get_model(NUM_CLASSES).to(DEVICE)

# ── Step 2: client_fn loads only its own indices from disk ──────────────────
def client_fn(context):
    cid = int(context.node_config["partition-id"])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    train_indices = np.load(f"cache/train_{cid}.npy")
    test_indices  = np.load(f"cache/test_{cid}.npy")

    train_set = Subset(full_dataset, train_indices.tolist())
    test_set  = Subset(full_dataset, test_indices.tolist())

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
            train(model, train_set, DEVICE,
                  global_weights=global_weights, mu=0.0)
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


# ── Step 3: run smoke test ───────────────────────────────────────────────────
strategy = FedAvgStrategy(model=GLOBAL_MODEL)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0.0},
    ray_init_args={"include_dashboard": False, "ignore_reinit_error": True}
)

print("\nPhase 4 smoke test passed.")