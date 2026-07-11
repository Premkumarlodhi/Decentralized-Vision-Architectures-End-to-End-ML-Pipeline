# client/worker.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import LOCAL_EPOCHS, BATCH_SIZE, LR


def train(model, dataset, device, global_weights=None, mu=0.0):
    # num_workers=0 to avoid multiprocessing issues on Windows and reduce memory overhead
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    criterion = nn.CrossEntropyLoss()
    model.train()

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss = criterion(model(x), y)

            if mu > 0 and global_weights is not None:
                prox = sum(
                    ((p - g.to(device)) ** 2).sum()
                    for p, g in zip(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        global_weights
                    )
                )
                loss += (mu / 2) * prox

            loss.backward()
            optimizer.step()


def evaluate(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            preds  = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

    return correct / total


def compress_delta(new_weights, global_weights, top_k=0.3):
    deltas = [
        (n.cpu() - g.cpu())
        for n, g in zip(new_weights, global_weights)
    ]

    flat      = torch.cat([d.flatten() for d in deltas])
    k         = max(1, int(top_k * flat.numel()))
    threshold = flat.abs().topk(k).values.min()

    sparse_deltas = []
    for d in deltas:
        mask = d.abs() >= threshold
        sparse_deltas.append(d * mask)

    non_zero   = sum((d != 0).sum().item() for d in sparse_deltas)
    bytes_sent = non_zero * 4

    return sparse_deltas, bytes_sent