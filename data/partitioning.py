# data/partitioning.py

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
from config import DATA_DIR, NUM_CLIENTS, ALPHA


def get_client_datasets(alpha=ALPHA, num_clients=NUM_CLIENTS):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    num_classes  = len(full_dataset.classes)
    targets      = np.array(full_dataset.targets)

    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_idx = np.where(targets == class_id)[0]
        np.random.shuffle(class_idx)

        proportions = np.random.dirichlet([alpha] * num_clients)

        splits      = (proportions * len(class_idx)).astype(int)
        splits[-1]  = len(class_idx) - splits[:-1].sum()  # fix rounding

        start = 0
        for client_id, count in enumerate(splits):
            client_indices[client_id].extend(
                class_idx[start:start + count].tolist()
            )
            start += count

    client_datasets = []
    for indices in client_indices:
        np.random.shuffle(indices)
        split    = int(0.8 * len(indices))
        train    = Subset(full_dataset, indices[:split])
        test     = Subset(full_dataset, indices[split:])
        client_datasets.append((train, test))

    return client_datasets, num_classes