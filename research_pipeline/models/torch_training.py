from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import resolve_path
from research_pipeline.data.tensors import load_landmark_npz
from research_pipeline.features.preprocessing import preprocess_dual_view
from research_pipeline.labels import TARGET_LABELS, label_to_index
from research_pipeline.models.artifacts import save_artifact
from research_pipeline.models.tcn import TCNConfig, build_tcn, require_torch
from research_pipeline.utils.errors import SchemaError
from research_pipeline.utils.random import set_global_seed


def _load_arrays(manifest_path: str | Path, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    records = read_jsonl(manifest_path)
    base_dir = Path(manifest_path).parent
    features: list[np.ndarray] = []
    labels: list[int] = []
    for record in records:
        if not record.tensor_path:
            raise SchemaError(f"Record '{record.sample_id}' has no tensor_path.")
        tensor = load_landmark_npz(resolve_path(record.tensor_path, base_dir))
        sequence = preprocess_dual_view(tensor, target_length=target_length)
        features.append(sequence.features)
        labels.append(label_to_index(record.target_label))
    if not features:
        raise SchemaError("Cannot train TCN on an empty manifest.")
    return np.stack(features).astype(np.float32), np.array(labels, dtype=np.int64)


def _stratified_validation_indices(
    labels: np.ndarray,
    *,
    validation_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if validation_split <= 0.0:
        indices = np.arange(labels.shape[0])
        return indices, np.array([], dtype=np.int64)
    if validation_split >= 1.0:
        raise SchemaError("validation_split must be in [0, 1).")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        if class_indices.shape[0] <= 1:
            train_indices.extend(class_indices.tolist())
            continue
        validation_count = max(1, int(round(class_indices.shape[0] * validation_split)))
        validation_count = min(validation_count, class_indices.shape[0] - 1)
        validation_indices.extend(class_indices[:validation_count].tolist())
        train_indices.extend(class_indices[validation_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return np.array(train_indices, dtype=np.int64), np.array(validation_indices, dtype=np.int64)


def _make_loader(
    torch,
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    balanced_sampler: bool = False,
):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np))
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if balanced_sampler and shuffle:
        class_counts = np.bincount(y_np, minlength=len(TARGET_LABELS)).astype(np.float64)
        sample_weights = 1.0 / np.maximum(class_counts[y_np], 1.0)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=int(y_np.shape[0]),
            replacement=True,
            generator=generator,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        generator=generator if shuffle and sampler is None else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _evaluate_loader(torch, model, criterion, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.detach().cpu()) * int(batch_y.shape[0])
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().detach().cpu())
            total += int(batch_y.shape[0])
    return total_loss / max(1, total), total_correct / max(1, total)


def _make_criterion(
    torch,
    nn,
    class_weights: np.ndarray,
    device,
    *,
    focal_gamma: float,
    label_smoothing: float,
):
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    smoothing = max(0.0, float(label_smoothing))
    if focal_gamma <= 0.0:
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)

    class FocalCrossEntropy(nn.Module):
        def __init__(self):
            super().__init__()
            self.gamma = float(focal_gamma)

        def forward(self, logits, target):
            weighted_ce = torch.nn.functional.cross_entropy(
                logits,
                target,
                weight=weights,
                reduction="none",
                label_smoothing=smoothing,
            )
            plain_ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
            pt = torch.exp(-plain_ce).clamp(1e-6, 1.0)
            return (((1.0 - pt) ** self.gamma) * weighted_ce).mean()

    return FocalCrossEntropy()


def _augment_batch(torch, batch_x, augmentation: dict):
    if not augmentation:
        return batch_x

    x = batch_x
    noise_sigma = float(augmentation.get("noise_sigma", 0.0))
    if noise_sigma > 0.0:
        x = x + torch.randn_like(x) * noise_sigma

    scale_std = float(augmentation.get("scale_std", 0.0))
    if scale_std > 0.0:
        factors = 1.0 + torch.randn((x.shape[0], 1, 1), device=x.device, dtype=x.dtype) * scale_std
        x = x * factors

    feature_dropout = float(augmentation.get("feature_dropout", 0.0))
    if feature_dropout > 0.0:
        keep = torch.rand_like(x) >= feature_dropout
        x = x * keep.to(dtype=x.dtype)

    temporal_shift = int(augmentation.get("temporal_shift", 0))
    if temporal_shift > 0:
        shifts = torch.randint(-temporal_shift, temporal_shift + 1, (x.shape[0],), device=x.device)
        if torch.any(shifts != 0):
            x = x.clone()
            for index, shift in enumerate(shifts.tolist()):
                if shift:
                    x[index] = torch.roll(x[index], shifts=int(shift), dims=0)

    time_mask_prob = float(augmentation.get("time_mask_prob", 0.0))
    time_mask_max_width = int(augmentation.get("time_mask_max_width", 0))
    if time_mask_prob > 0.0 and time_mask_max_width > 0:
        x = x.clone()
        width_limit = min(time_mask_max_width, x.shape[1])
        for index in range(x.shape[0]):
            if float(torch.rand((), device=x.device)) < time_mask_prob:
                width = int(torch.randint(1, width_limit + 1, (), device=x.device))
                start = int(torch.randint(0, x.shape[1] - width + 1, (), device=x.device))
                x[index, start : start + width, :] = 0.0
    return x


def train_tcn(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 13,
    target_length: int = 32,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    validation_manifest_path: str | Path | None = None,
    validation_split: float = 0.0,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    channels: tuple[int, ...] | None = None,
    kernel_size: int = 3,
    dropout: float = 0.15,
    pooling: str = "avg",
    balanced_sampler: bool = False,
    focal_gamma: float = 0.0,
    label_smoothing: float = 0.0,
    augmentation: dict | None = None,
) -> dict:
    torch, nn = require_torch()
    set_global_seed(seed)

    x_np, y_np = _load_arrays(manifest_path, target_length)
    validation_source = "none"
    if validation_manifest_path:
        train_x_np, train_y_np = x_np, y_np
        val_x_np, val_y_np = _load_arrays(validation_manifest_path, target_length)
        validation_source = str(validation_manifest_path)
    else:
        train_indices, validation_indices = _stratified_validation_indices(
            y_np,
            validation_split=validation_split,
            seed=seed,
        )
        train_x_np, train_y_np = x_np[train_indices], y_np[train_indices]
        val_x_np, val_y_np = x_np[validation_indices], y_np[validation_indices]
        validation_source = "stratified_split" if validation_indices.size else "none"

    config = TCNConfig(
        input_dim=int(x_np.shape[-1]),
        num_classes=len(TARGET_LABELS),
        channels=tuple(channels) if channels else TCNConfig().channels,
        kernel_size=int(kernel_size),
        dropout=float(dropout),
        pooling=str(pooling),
    )
    model = build_tcn(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_counts = np.bincount(train_y_np, minlength=len(TARGET_LABELS)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = _make_criterion(
        torch,
        nn,
        class_weights,
        device,
        focal_gamma=float(focal_gamma),
        label_smoothing=float(label_smoothing),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    loader = _make_loader(
        torch,
        train_x_np,
        train_y_np,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        balanced_sampler=balanced_sampler,
    )
    validation_loader = None
    if val_x_np.size:
        validation_loader = _make_loader(
            torch,
            val_x_np,
            val_y_np,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )

    history: list[dict[str, float]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_state_dict: dict | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    stopped_early = False
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_x = _augment_batch(torch, batch_x, augmentation or {})
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * int(batch_y.shape[0])
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().detach().cpu())
            total += int(batch_y.shape[0])
        scheduler.step()
        epoch_report = {
            "epoch": float(epoch),
            "loss": total_loss / max(1, total),
            "train_accuracy": total_correct / max(1, total),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        if validation_loader is not None:
            validation_loss, validation_accuracy = _evaluate_loader(torch, model, criterion, validation_loader, device)
            epoch_report["validation_loss"] = validation_loss
            epoch_report["validation_accuracy"] = validation_accuracy
            if validation_loss < best_validation_loss - early_stopping_min_delta:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state_dict = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                stale_epochs = 0
            else:
                stale_epochs += 1
            if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
                stopped_early = True
                history.append(epoch_report)
                break
        history.append(epoch_report)

    if best_state_dict is not None:
        state_dict = best_state_dict
    else:
        state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    artifact = {
        "model_type": "c1t_tcn_torch",
        "labels": list(TARGET_LABELS),
        "target_length": target_length,
        "seed": seed,
        "tcn_config": asdict(config),
        "state_dict": state_dict,
        "history": history,
        "training": {
            "train_samples": int(train_y_np.shape[0]),
            "validation_samples": int(val_y_np.shape[0]),
            "validation_source": validation_source,
            "validation_split": float(validation_split),
            "early_stopping_patience": int(early_stopping_patience),
            "early_stopping_min_delta": float(early_stopping_min_delta),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation_loss) if best_state_dict is not None else None,
            "stopped_early": stopped_early,
            "epochs_completed": len(history),
            "balanced_sampler": bool(balanced_sampler),
            "focal_gamma": float(focal_gamma),
            "label_smoothing": float(label_smoothing),
            "augmentation": dict(augmentation or {}),
        },
    }
    save_artifact(output_path, artifact)
    return artifact
