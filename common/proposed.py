"""Train and evaluate the proposed IPM/coefficient dual-branch classifier.

Inputs are precomputed per-frame IPM, coefficient, and block-map NumPy arrays,
plus train/validation/test label-map JSON files for each fold.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


def make_features(ipm, coeff, block_map):
    """Return the 37-channel IPM and 67-channel coefficient model inputs."""
    if ipm.ndim != 2 or ipm.shape != coeff.shape or ipm.shape != block_map.shape:
        raise ValueError(
            "IPM, coefficient, and block maps must have the same 2D shape")
    if not np.issubdtype(ipm.dtype, np.integer) or np.any(
            (ipm < 0) | (ipm > 34)):
        raise ValueError("IPM values must be integers from 0 to 34")

    height, width = ipm.shape
    x = np.broadcast_to(
        np.linspace(
            0, 1, width, dtype=np.float32), (height, width))
    y = np.broadcast_to(np.linspace(0, 1, height, dtype=np.float32)[:, None],
                        (height, width))
    positions = np.stack((x, y))

    ipm_onehot = np.eye(35, dtype=np.float32)[ipm].transpose(2, 0, 1)
    ipm_features = np.concatenate((ipm_onehot, positions), axis=0)

    # Preserve the original 64 coefficient bins and block-map channel.
    coeff_indices = np.clip(coeff, -32, 31).astype(np.int64) + 32
    coeff_onehot = np.eye(
        64, dtype=np.float32)[coeff_indices].transpose(
        2, 0, 1)
    coeff_features = np.concatenate(
        (coeff_onehot, block_map.astype(np.float32)[None], positions), axis=0
    )
    return ipm_features, coeff_features


class IPMCoeffDataset(Dataset):
    """Read one cover or stego frame from each label-map entry."""

    def __init__(self, data_root, label_map_path):
        self.data_root = Path(data_root)
        with Path(label_map_path).open(encoding="utf-8") as file:
            self.samples = list(json.load(file).items())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_id, label = self.samples[index]
        if "_frame_" not in frame_id:
            raise ValueError(f"Invalid frame ID: {frame_id}")
        segment, frame = frame_id.rsplit("_frame_", 1)
        is_stego = frame.endswith("_stego")
        if is_stego:
            frame = frame[:-6]
        suffix = "_stego" if is_stego else ""
        stem = self.data_root / segment / f"frame_{frame}{suffix}"

        ipm = np.load(f"{stem}_ipm.npy", allow_pickle=False)
        coeff = np.load(f"{stem}_coeff.npy", allow_pickle=False)
        block_map = np.load(f"{stem}_block.npy", allow_pickle=False)
        ipm_features, coeff_features = make_features(ipm, coeff, block_map)
        return (
            torch.from_numpy(ipm_features),
            torch.from_numpy(coeff_features),
            torch.tensor(int(label), dtype=torch.long),
        )


def split_paths(split_dir, fold):
    """Return train, validation, and test label-map paths for one fold."""
    root = Path(split_dir)
    return tuple(root / f"label_map_{part}_split{fold}.json"
                 for part in ("train", "val", "test"))


def _evaluate(model, loader, criterion, device):
    model.eval()
    losses, labels, probabilities = [], [], []
    with torch.no_grad():
        for ipm, coeff, target in loader:
            logits = model(ipm.to(device), coeff.to(device))
            losses.append(criterion(logits, target.to(device)).item())
            labels.extend(target.tolist())
            probabilities.extend(
                torch.softmax(
                    logits, dim=1)[
                    :, 1].cpu().tolist())
    if not labels:
        raise ValueError("The evaluation split is empty")
    predictions = [int(probability >= 0.5) for probability in probabilities]
    accuracy = sum(
        pred == label for pred,
        label in zip(
            predictions,
            labels)) / len(labels)
    auc = roc_auc_score(
        labels, probabilities) if len(
        set(labels)) == 2 else None
    return {"loss": float(np.mean(losses)), "accuracy": accuracy, "auc": auc}


def train_fold(data_root, split_dir, fold=1, output_dir="output", epochs=50,
               batch_size=16, learning_rate=1e-4, device="auto", model_factory=None):
    """Train one fold, save the best checkpoint, and report test metrics."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                          else "cpu" if device == "auto" else device)
    if model_factory is None:
        from model import DualBranchEnsemble
        model_factory = DualBranchEnsemble

    datasets = [IPMCoeffDataset(data_root, path)
                for path in split_paths(split_dir, fold)]
    if any(len(dataset) == 0 for dataset in datasets):
        raise ValueError(
            "Train, validation, and test splits must all be nonempty")
    train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=batch_size)
    test_loader = DataLoader(datasets[2], batch_size=batch_size)

    model = model_factory().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_state = float("inf"), None

    for epoch in range(epochs):
        model.train()
        for ipm, coeff, labels in train_loader:
            optimizer.zero_grad()
            logits = model(ipm.to(device), coeff.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
        metrics = _evaluate(model, val_loader, criterion, device)
        print(f"Fold {fold} epoch {epoch + 1}/{epochs}: "
              f"validation loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}")
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / f"best_model_fold{fold}.pth")
    result = _evaluate(model, test_loader, criterion, device)
    print(
        f"Fold {fold} test accuracy={result['accuracy']:.4f}, AUC={result['auc']}")
    return result


def load_config(config_path):
    path = Path(config_path)
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    for key in ("data_root", "split_dir"):
        if not config.get(key):
            raise ValueError(f"config.json requires {key}")
    for key in ("data_root", "split_dir", "output_dir"):
        if key in config:
            value = Path(config[key])
            config[key] = value if value.is_absolute() else path.parent / value
    return config


def run(factory, config_path):
    """Run one configured experiment with the supplied model factory."""
    train_fold(**load_config(config_path), model_factory=factory)
