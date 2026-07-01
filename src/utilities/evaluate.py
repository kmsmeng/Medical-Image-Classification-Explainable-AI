import torch
import numpy as np

def evaluate(model, loader, criterion, device, classes:None):
    """
    Run evaluation on val or test set

    Returns:
        avg_loss : float
        aucs     : dict with per-class AUC-ROC and 'mean'
    """

    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_labels = []
    all_probs = []

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            with torch.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, labels)
            
            probs = torch.sigmoid(logits)

            total_loss += loss.item()

            all_probs.append(probs.cpu().numpy())

            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f'    Batch: {batch_idx+1}/{len(loader)} | loss: {loss.item():.4f}')


            # all_probs = np.concatenate(all)

        return probs