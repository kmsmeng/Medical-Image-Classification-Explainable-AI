import torch

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """
    Run one training epoch with mixed precision

    Returns:
        avg_loss : float - mean loss over all batches
    """

    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(images) # (B, 14)
            loss = criterion(logits, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 200 == 0:
            print(f'    Batch {batch_idx + 1}/{len(loader)} | Loss: {loss.item():.4f}')

    return total_loss / num_batches