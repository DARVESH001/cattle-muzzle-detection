import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.network import SiameseNetwork, ContrastiveLoss
from model.dataset import CattleMuzzleDataset, SingleImageDataset, get_transforms, split_cattle_ids


def compute_accuracy(model, dataloader, device, threshold=config.MARGIN / 2):
    """Compute pair classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            emb1, emb2 = model(img1, img2)
            distance = F.pairwise_distance(emb1, emb2)
            # Predict same cattle if distance < threshold
            pred = (distance < threshold).float()
            correct += (pred == label).sum().item()
            total += label.size(0)
    return correct / total if total > 0 else 0.0


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Split data
    train_ids, val_ids, test_ids = split_cattle_ids(args.dataset_path)

    # Datasets
    train_dataset = CattleMuzzleDataset(
        args.dataset_path, train_ids,
        transform=get_transforms(is_training=True),
        pairs_per_epoch=args.pairs_per_epoch,
    )
    val_dataset = CattleMuzzleDataset(
        args.dataset_path, val_ids,
        transform=get_transforms(is_training=False),
        pairs_per_epoch=500,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # Model
    model = SiameseNetwork(embedding_dim=config.EMBEDDING_DIM).to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Freeze backbone initially
    for param in model.embedding_net.features.parameters():
        param.requires_grad = False
    print(f"Backbone frozen for first {config.FREEZE_EPOCHS} epochs")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after FREEZE_EPOCHS
        if epoch == config.FREEZE_EPOCHS + 1:
            for param in model.embedding_net.features.parameters():
                param.requires_grad = True
            # Lower LR for backbone fine-tuning
            optimizer = torch.optim.Adam([
                {'params': model.embedding_net.features.parameters(), 'lr': args.lr * 0.1},
                {'params': model.embedding_net.head.parameters(), 'lr': args.lr},
            ], weight_decay=config.WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            print("Backbone unfrozen - fine-tuning with lower LR")

        # Training
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        elapsed = time.time() - start_time

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                emb1, emb2 = model(img1, img2)
                loss = criterion(emb1, emb2, label)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = compute_accuracy(model, val_loader, device)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
            }, config.MODEL_PATH)
            print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")

    print(f"\nTraining complete. Best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    checkpoint = torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = CattleMuzzleDataset(
        args.dataset_path, test_ids,
        transform=get_transforms(is_training=False),
        pairs_per_epoch=1000,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=config.NUM_WORKERS,
    )

    test_acc = compute_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2%}")

    # Compute reference embeddings for muzzle detection
    print("\n--- Computing reference embeddings for muzzle detection ---")
    save_reference_embeddings(model, args.dataset_path, train_ids, device)

    print("\nDone! Model saved to:", config.MODEL_PATH)


def save_reference_embeddings(model, dataset_path, cattle_ids, device):
    """Save average embeddings from training cattle for muzzle detection."""
    import numpy as np
    model.eval()
    transform = get_transforms(is_training=False)
    all_embeddings = []

    dataset = SingleImageDataset(dataset_path, cattle_ids, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=config.NUM_WORKERS)

    with torch.no_grad():
        for imgs, _, _ in loader:
            imgs = imgs.to(device)
            embs = model.get_embedding(imgs)
            all_embeddings.append(embs.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    # Save mean embedding as reference for "is this a muzzle?" detection
    mean_embedding = np.mean(all_embeddings, axis=0)
    ref_path = os.path.join(config.SAVED_MODELS_DIR, "reference_embeddings.npy")
    np.save(ref_path, mean_embedding)
    print(f"Saved reference embedding ({all_embeddings.shape[0]} images) to {ref_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cattle Muzzle Siamese Network")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--margin", type=float, default=config.MARGIN)
    parser.add_argument("--dataset-path", type=str, default=config.DATASET_PATH)
    parser.add_argument("--pairs-per-epoch", type=int, default=config.PAIRS_PER_EPOCH)
    args = parser.parse_args()

    train(args)
