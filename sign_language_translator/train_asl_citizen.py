"""
Train LSTM on ASL Citizen

Run processor first:
    python3 asl_citizen_processor.py --top-n 20

Then train:
    python3 train_asl_citizen.py --epochs 50 --hidden-size 256 --layers 4
"""

import torch
import pandas as pd
import json
from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from lstm_model import Video_LSTM
from training   import train_model, visualize_results


# Dataset

class ASLCitizenDataset(Dataset):
    def __init__(self, directory="final_merged", partition="train"):
        self.data_dir = Path(directory) / partition
        metadata      = pd.read_csv(Path(directory) / "glosses.csv")
        self.metadata = metadata[metadata["partition"] == partition].reset_index(drop=True)
        print(f"  {partition}: {len(self.metadata)} videos")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        video = torch.load(self.data_dir / f"{index}.pt")
        label = int(self.metadata.iloc[index]["label"])
        return video, label


def get_dataloaders(processed_dir="final_merged"):
    train_ds     = ASLCitizenDataset(processed_dir, "train")
    val_ds       = ASLCitizenDataset(processed_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
    return train_loader, val_loader


def get_class_weights(processed_dir, num_classes, device):
    """
    Compute inverse-frequency class weights so rare classes get
    higher loss penalty — prevents the model collapsing to majority classes.
    """
    metadata   = pd.read_csv(Path(processed_dir) / "glosses.csv")
    train_meta = metadata[metadata["partition"] == "train"]
    counts     = Counter(train_meta["label"].tolist())
    weights    = torch.tensor(
        [1.0 / (counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32
    )
    weights = weights / weights.sum() * num_classes  # normalize
    return weights.to(device)


# ─── Training ─────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("checkpoints")

def save_checkpoint(epoch, model, optimizer, results, model_name):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"{model_name}_epoch{epoch:03d}.pt"
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "results":         results,          # accumulated loss/acc history
    }, path)
    print(f"  [checkpoint] saved -> {path}")


def load_latest_checkpoint(model_name, model, optimizer):
    """
    Finds the highest-epoch checkpoint for this model_name and loads it.
    Returns (start_epoch, results) or (0, default_results) if none found.
    """
    checkpoints = sorted(CHECKPOINT_DIR.glob(f"{model_name}_epoch*.pt"))
    if not checkpoints:
        return 0, {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    latest = checkpoints[-1]
    print(f"  [checkpoint] resuming from {latest}")
    ckpt = torch.load(latest, weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    return ckpt["epoch"] + 1, ckpt["results"]   # start AFTER the saved epoch


def train_asl_citizen(
    processed_dir = "final_merged",
    epochs        = 50,
    lr            = 1e-4,
    n_layers      = 4,
    hidden_size   = 256,
    dropout       = 0.5,
    model_name    = "asl_citizen",
    resume        = False,          # ← new
):
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        print(f"Processed data not found at '{processed_dir}'")
        return

    cfg         = pd.read_csv(processed_path / "config.csv").iloc[0]
    feature_dim = int(cfg["feature_dim"])
    num_classes = int(cfg["num_classes"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_loader, val_loader = get_dataloaders(processed_dir)

    model = Video_LSTM(
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers=n_layers,
        num_classes=num_classes,
        input_size=feature_dim,
    )

    class_weights = get_class_weights(processed_dir, num_classes, device)
    criterion     = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ── Resume or start fresh ──────────────────────────────────────────────
    if resume:
        start_epoch, all_results = load_latest_checkpoint(model_name, model, optimizer)
    else:
        start_epoch  = 0
        all_results  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("=" * 70)
    print("TRAINING ON ASL CITIZEN")
    print("=" * 70)
    print(f"  Resuming from epoch : {start_epoch} / {epochs}")
    print(f"  Device              : {device}")
    print()

    # ── Epoch-by-epoch loop with checkpointing ────────────────────────────
    for epoch in range(start_epoch, epochs):
        print(f"── Epoch {epoch + 1} / {epochs} ──────────────────────────────")

        # train_model runs exactly one epoch
        epoch_results = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            save_prefix=model_name,
        )

        # Accumulate results (assuming train_model returns dict of lists)
        for key in all_results:
            all_results[key].extend(epoch_results.get(key, []))

        save_checkpoint(epoch, model, optimizer, all_results, model_name)

    # ── Done ──────────────────────────────────────────────────────────────
    visualize_results(all_results, save_prefix=model_name)
    print("\nTraining complete!")
    print(f"   Model  -> saved_models/{model_name}_fc_model.pth")
    print(f"   Plots  -> saved_plots/")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default="final_merged")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--hidden-size",   type=int,   default=256)
    p.add_argument("--layers",        type=int,   default=4)
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--model-name",    default="asl_citizen")
    p.add_argument("--resume",        action="store_true")   # ← new
    args = p.parse_args()

    train_asl_citizen(
        processed_dir=args.processed_dir,
        epochs=args.epochs,
        lr=args.lr,
        n_layers=args.layers,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        model_name=args.model_name,
        resume=args.resume,             # ← new
    )
