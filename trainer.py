"""Minimal GPT-style trainer — deliberately MISSING gradient clipping.

antar should detect: AdamW + cosine schedule + mixed precision present,
but NO clip_grad_norm_ before optimizer.step() → training instability gap.
"""
import torch

model = torch.nn.Linear(128, 128)  # placeholder for a transformer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
scaler = torch.cuda.amp.GradScaler()  # mixed precision


def train_step(batch):
    x, y = batch
    optimizer.zero_grad()
    with torch.autocast(device_type="cpu"):
        logits = model(x)
        loss = torch.nn.functional.mse_loss(logits, y)
    scaler.scale(loss).backward()
    # BUG: no torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) here
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.item()
