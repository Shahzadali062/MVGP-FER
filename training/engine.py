"""One pass through training or evaluation data."""

import torch
from torch.nn import functional as F


def run_epoch(model, loader, device, optimizer=None, scaler=None, amp=False):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = total_ok = total_count = 0
    for images, labels, geometry, cues, valid in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        geometry = geometry.to(device, non_blocking=True)
        cues = cues.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(images, geometry, cues, valid)
                logits = output["logits"]
                loss = F.cross_entropy(logits, labels)
            if is_training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        batch = labels.numel()
        total_loss += loss.detach().item() * batch
        total_ok += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch
    if total_count == 0:
        raise ValueError("Cannot process an empty dataset")
    return {"loss": total_loss / total_count, "accuracy": total_ok / total_count,
            "samples": total_count}
