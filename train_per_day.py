import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from efficient_per_day_data_loader import create_walk_forward_splits, get_walk_forward_dataloaders
from per_day_copy_trading_model import PerDayCopyTradingModel


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def move_batch_to_device(batch, device):
    """Move batch to device"""
    moved_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved_batch[k] = v.to(device)
        elif isinstance(v, list):
            if k == 'daily_sequences':
                moved_batch[k] = [tensor.to(device) for tensor in v]
            else:
                moved_batch[k] = v
        else:
            moved_batch[k] = v
    return moved_batch


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = move_batch_to_device(batch, device)
        predictions = model(batch)
        labels = batch['label']
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary_preds = (all_preds > 0.5).astype(int)

    metrics = {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': accuracy_score(all_labels, binary_preds),
        'precision': precision_score(all_labels, binary_preds, zero_division=0),
        'recall': recall_score(all_labels, binary_preds, zero_division=0),
        'f1': f1_score(all_labels, binary_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    }
    return metrics


def evaluate_with_predictions(model, dataloader, criterion, device):
    """Evaluate model and return predictions with transaction IDs"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_transaction_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = move_batch_to_device(batch, device)
            predictions = model(batch)
            labels = batch['label']
            loss = criterion(predictions, labels)

            total_loss += loss.item() * len(labels)
            all_preds.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_transaction_ids.extend(batch['transaction_id'])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary_preds = (all_preds > 0.5).astype(int)

    metrics = {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': accuracy_score(all_labels, binary_preds),
        'precision': precision_score(all_labels, binary_preds, zero_division=0),
        'recall': recall_score(all_labels, binary_preds, zero_division=0),
        'f1': f1_score(all_labels, binary_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5,
        'predictions': all_preds,
        'binary_predictions': binary_preds,
        'labels': all_labels,
        'transaction_ids': all_transaction_ids
    }
    return metrics


def save_predictions_csv(metrics, output_dir, test_year, label_period):
    """Save predictions to CSV with transaction IDs, probabilities, and labels"""
    df = pd.DataFrame({
        'transaction_id': metrics['transaction_ids'],
        'true_label': metrics['labels'],
        'predicted_probability': metrics['predictions'],
        'predicted_label': metrics['binary_predictions']
    })

    csv_path = os.path.join(output_dir, f'predictions_{test_year}_{label_period}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")
    return csv_path


def train_walk_forward_model(args, walk_forward_split):
    """Train model for a single walk-forward split"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_year = walk_forward_split['test_year']

    print(f"\n{'=' * 80}")
    print(f"TRAINING MODEL FOR TEST YEAR {test_year}")
    print(f"Training Period: {walk_forward_split['train_period']}")
    print(f"{'=' * 80}")

    # Create dataloaders for this split
    train_loader, val_loader, test_loader = get_walk_forward_dataloaders(
        args.dataset_path,
        label_period=args.label_period,
        batch_size=args.batch_size,
        walk_forward_split=walk_forward_split
    )

    # Get sample for model dimensions
    sample_batch = next(iter(train_loader))
    daily_token_dim = sample_batch['daily_sequences'][0].shape[1] if sample_batch['daily_sequences'][0].shape[
                                                                         0] > 0 else 392

    # Create model
    model = PerDayCopyTradingModel(
        cp_input_dim=sample_batch['target_features'].shape[1],
        cp_hidden_dim=args.cp_hidden_dim,
        cp_embed_dim=args.cp_embed_dim,
        stock_feature_dim=5,
        daily_token_dim=daily_token_dim,
        transformer_hidden_dim=args.transformer_hidden_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        fusion_dim=args.fusion_dim,
        prediction_hidden_dim=args.prediction_hidden_dim,
        dropout=args.dropout
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate * 5,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_auc = 0

    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['accuracy'])

        # Validate
        val_metrics = evaluate_with_predictions(model, val_loader, criterion, device)
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['accuracy'])

        print(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")

        scheduler.step()

        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = model.state_dict().copy()

        # Early stopping
        if early_stopping(val_metrics['loss'], model):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model and test
    model.load_state_dict(best_model_state)
    test_metrics = evaluate_with_predictions(model, test_loader, criterion, device)

    print(f"\nTest Results - Year {test_year}:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")

    return {
        'test_year': test_year,
        'test_metrics': test_metrics,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    }


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset and creating walk-forward splits...")

    # Load dataset to create splits
    import pickle
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # Create walk-forward splits
    walk_forward_splits = create_walk_forward_splits(dataset)

    if not walk_forward_splits:
        print("No valid walk-forward splits found!")
        return

    # Store all results
    all_results = []
    all_predictions = []

    # Train model for each walk-forward split
    for split in walk_forward_splits:
        try:
            result = train_walk_forward_model(args, split)
            all_results.append(result)

            # Save predictions CSV
            csv_path = save_predictions_csv(
                result['test_metrics'],
                args.output_dir,
                result['test_year'],
                args.label_period
            )
            all_predictions.append(csv_path)

        except Exception as e:
            print(f"Error training model for year {split['test_year']}: {e}")
            continue

    # Create summary results
    if all_results:
        summary_df = pd.DataFrame([
            {
                'test_year': result['test_year'],
                'accuracy': result['test_metrics']['accuracy'],
                'precision': result['test_metrics']['precision'],
                'recall': result['test_metrics']['recall'],
                'f1': result['test_metrics']['f1'],
                'auc': result['test_metrics']['auc'],
                'test_samples': len(result['test_metrics']['labels'])
            }
            for result in all_results
        ])

        summary_path = os.path.join(args.output_dir, f'walk_forward_summary_{args.label_period}.csv')
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'=' * 80}")
        print("WALK-FORWARD VALIDATION SUMMARY")
        print(f"{'=' * 80}")
        print(summary_df.to_string(index=False))
        print(f"\nOverall Average AUC: {summary_df['auc'].mean():.4f}")
        print(f"Overall Average F1: {summary_df['f1'].mean():.4f}")
        print(f"Summary saved to: {summary_path}")

        print("\nPrediction files created:")
        for pred_file in all_predictions:
            print(f"  {pred_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-Forward Copy-Trading Prediction")

    # Data arguments
    parser.add_argument("--dataset_path", type=str, default="efficient_per_day_trading_dataset.pkl")
    parser.add_argument("--label_period", type=str, default="1M", choices=["1M", "3M", "6M"])

    # Model arguments
    parser.add_argument("--cp_hidden_dim", type=int, default=256)
    parser.add_argument("--cp_embed_dim", type=int, default=128)
    parser.add_argument("--transformer_hidden_dim", type=int, default=256)
    parser.add_argument("--transformer_layers", type=int, default=4)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--prediction_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)  # Reduced for walk-forward
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--output_dir", type=str, default="walk_forward_results")

    args = parser.parse_args()
    main(args)
