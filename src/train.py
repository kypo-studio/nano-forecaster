from __future__ import annotations

import copy
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def limited(dataset, limit):
    return dataset if limit is None or len(dataset) <= limit else Subset(dataset, range(limit))


def train_model(model, train_dataset, val_dataset, config, seed: int):
    set_seed(seed)
    device = choose_device(config["device"])
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    loss_fn = nn.MSELoss()
    history = {"train": [], "val": []}
    best_loss, best_state, stale = float("inf"), None, 0
    started = time.perf_counter()
    for _ in range(config["epochs"]):
        model.train()
        total = count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x); count += len(x)
        history["train"].append(total / count)
        model.eval(); total = count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                total += loss.item() * len(x); count += len(x)
        val_loss = total / count
        history["val"].append(val_loss)
        if val_loss < best_loss:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    return history, time.perf_counter() - started, device


def predict(model, dataset, batch_size: int, device: torch.device):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    outputs, targets, inputs = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            outputs.append(model(x.to(device)).cpu().numpy())
            targets.append(y.numpy()); inputs.append(x.numpy())
    return np.concatenate(outputs), np.concatenate(targets), np.concatenate(inputs)

