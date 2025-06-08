import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import math
from datetime import datetime


def get_range_midpoint(range_str):
    """Convert range string to log-transformed midpoint"""
    try:
        parts = range_str.split(' - ')
        if len(parts) == 2:
            low, high = float(parts[0]), float(parts[1])
            midpoint = (low + high) / 2
        else:
            midpoint = float(parts[0])
        return math.log1p(midpoint)
    except:
        return 0.0


def create_walk_forward_splits(dataset, start_year=2014, end_year=2025):
    """Create walk-forward validation splits based on filing dates"""

    # Sort all transactions by filing date
    dated_transactions = []
    for tid in dataset.keys():
        if tid == '__embeddings__':
            continue
        filed_date = dataset[tid]['current_transaction'].get('Filed')
        if filed_date and not pd.isna(filed_date):
            filed_dt = pd.to_datetime(filed_date)
            dated_transactions.append((filed_dt, tid))

    dated_transactions.sort()

    # Create walk-forward splits
    walk_forward_splits = []

    for test_year in range(2020, end_year + 1):  # Start testing from 2020
        # Training window: 6 years before test year
        train_start = test_year - 6
        train_end = test_year - 1

        print(f"\nWalk-Forward Split for testing {test_year}:")
        print(f"Training on: {train_start}-{train_end}")

        # Filter transactions by filing date
        train_ids = []
        val_ids = []
        test_ids = []

        for filed_dt, tid in dated_transactions:
            year = filed_dt.year

            if train_start <= year <= train_end:
                # Split train/val within training period (80/20)
                if year <= train_end - 1 or (year == train_end and filed_dt.month <= 9):
                    train_ids.append(tid)
                else:
                    val_ids.append(tid)
            elif year == test_year:
                test_ids.append(tid)

        if len(train_ids) > 0 and len(test_ids) > 0:
            split_info = {
                'test_year': test_year,
                'train_period': f"{train_start}-{train_end}",
                'train_ids': train_ids,
                'val_ids': val_ids if len(val_ids) > 0 else train_ids[-len(train_ids) // 5:],  # Fallback to last 20%
                'test_ids': test_ids,
                'train_count': len(train_ids),
                'val_count': len(val_ids) if len(val_ids) > 0 else len(train_ids) // 5,
                'test_count': len(test_ids)
            }

            # Ensure validation set exists
            if len(val_ids) == 0:
                # Take last 20% of training as validation
                val_size = len(train_ids) // 5
                split_info['val_ids'] = train_ids[-val_size:]
                split_info['train_ids'] = train_ids[:-val_size]
                split_info['train_count'] = len(split_info['train_ids'])
                split_info['val_count'] = len(split_info['val_ids'])

            walk_forward_splits.append(split_info)

            print(f"  Train: {split_info['train_count']} transactions")
            print(f"  Val: {split_info['val_count']} transactions")
            print(f"  Test: {split_info['test_count']} transactions")

    return walk_forward_splits


class EfficientPerDayTradingDataset(Dataset):
    def __init__(self, dataset_path, label_period='1M', transaction_ids=None, reference_scaler=None,
                 ablation_mode='full'):
        # Load dataset
        with open(dataset_path, 'rb') as f:
            self.full_dataset = pickle.load(f)

        # Extract embeddings
        self.embeddings_dict = self.full_dataset.pop('__embeddings__')

        # Fixed dimensions
        self.EMBEDDING_DIM = 384
        self.TRANSACTION_TYPE_DIM = 2
        self.TRADE_SIZE_DIM = 1
        self.CP_FEATURE_DIM = self.EMBEDDING_DIM + self.TRANSACTION_TYPE_DIM + self.TRADE_SIZE_DIM
        self.STOCK_DIM = 5
        # Adjust daily token dimension based on ablation mode
        if ablation_mode == 'target_cp_only':
            self.DAILY_TOKEN_DIM = 0  # No daily sequence data
        else:
            self.DAILY_TOKEN_DIM = self.STOCK_DIM + self.CP_FEATURE_DIM  # Full historical data

        # Use provided transaction IDs or filter by label availability
        if transaction_ids is not None:
            self.transaction_ids = transaction_ids
        else:
            label_key = f'Label_{label_period}'
            self.transaction_ids = [
                tid for tid in self.full_dataset.keys()
                if (tid in self.full_dataset and
                    'labels' in self.full_dataset[tid] and
                    self.full_dataset[tid]['labels'].get(label_key) is not None and
                    not pd.isna(self.full_dataset[tid]['labels'].get(label_key)))
            ]

        # Initialize or use reference scaler
        self.trade_size_scaler = reference_scaler if reference_scaler else MinMaxScaler()
        if reference_scaler is None:
            midpoints = []
            for tid in self.transaction_ids:
                current_tx = self.full_dataset[tid]['current_transaction']
                if 'Trade_Size_USD' in current_tx:
                    midpoints.append(get_range_midpoint(current_tx['Trade_Size_USD']))
            if midpoints:
                self.trade_size_scaler.fit(np.array(midpoints).reshape(-1, 1))

        self.label_period = label_period
        self.ablation_mode = ablation_mode
        print(f"Created dataset with {len(self.transaction_ids)} samples, ablation mode: {ablation_mode}")

    def _create_cp_features_robust(self, transaction_id):
        # Existing method unchanged
        try:
            if transaction_id in self.embeddings_dict:
                raw_embedding = self.embeddings_dict[transaction_id]
                if len(raw_embedding) == self.EMBEDDING_DIM:
                    embedding = torch.tensor(raw_embedding, dtype=torch.float32)
                elif len(raw_embedding) < self.EMBEDDING_DIM:
                    padded = raw_embedding + [0.0] * (self.EMBEDDING_DIM - len(raw_embedding))
                    embedding = torch.tensor(padded, dtype=torch.float32)
                else:
                    embedding = torch.tensor(raw_embedding[:self.EMBEDDING_DIM], dtype=torch.float32)
            else:
                embedding = torch.zeros(self.EMBEDDING_DIM, dtype=torch.float32)

            if transaction_id in self.full_dataset:
                tx_type = self.full_dataset[transaction_id]['current_transaction'].get('Transaction', '')
                trade_size = self.full_dataset[transaction_id]['current_transaction'].get('Trade_Size_USD', '0')
            else:
                tx_type = ''
                trade_size = '0'

            tx_onehot = torch.zeros(self.TRANSACTION_TYPE_DIM, dtype=torch.float32)
            if tx_type == 'Purchase':
                tx_onehot[0] = 1.0
            elif 'Sale' in tx_type:
                tx_onehot[1] = 1.0

            size_value = get_range_midpoint(trade_size)
            norm_size = self.trade_size_scaler.transform([[size_value]])[0][0]
            size_tensor = torch.tensor([norm_size], dtype=torch.float32)

            cp_features = torch.cat([embedding, tx_onehot, size_tensor])
            if cp_features.shape[0] != self.CP_FEATURE_DIM:
                correct_features = torch.zeros(self.CP_FEATURE_DIM, dtype=torch.float32)
                copy_len = min(cp_features.shape[0], self.CP_FEATURE_DIM)
                correct_features[:copy_len] = cp_features[:copy_len]
                cp_features = correct_features
            return cp_features
        except Exception as e:
            print(f"ERROR creating CP features for {transaction_id}: {e}")
            return torch.zeros(self.CP_FEATURE_DIM, dtype=torch.float32)

    def __len__(self):
        return len(self.transaction_ids)

    def __getitem__(self, idx):
        tid = self.transaction_ids[idx]
        transaction_data = self.full_dataset[tid]

        # Target CP features
        target_features = self._create_cp_features_robust(tid)

        # Build daily sequence only if not in target_cp_only mode
        if self.ablation_mode == 'target_cp_only':
            daily_sequence_tensor = torch.zeros((0, 0), dtype=torch.float32)
        else:
            stock_history = transaction_data.get('stock_history', [])
            cp_activity = transaction_data.get('cp_activity', {})
            daily_sequence = []

            for day_data in stock_history:
                # Stock features
                try:
                    stock_features = torch.tensor([
                        float(day_data.get('open', 0)),
                        float(day_data.get('high', 0)),
                        float(day_data.get('low', 0)),
                        float(day_data.get('close', 0)),
                        float(day_data.get('volume', 0))
                    ], dtype=torch.float32)
                    if stock_features.shape[0] != self.STOCK_DIM:
                        stock_features = torch.zeros(self.STOCK_DIM, dtype=torch.float32)
                except:
                    stock_features = torch.zeros(self.STOCK_DIM, dtype=torch.float32)

                # CP activity
                day_date = day_data['date']
                try:
                    if day_date in cp_activity and cp_activity[day_date]:
                        cp_ids_on_day = cp_activity[day_date]
                        cp_vectors = []
                        for cp_id in cp_ids_on_day:
                            cp_vec = self._create_cp_features_robust(cp_id)
                            cp_vectors.append(cp_vec)
                        if cp_vectors:
                            cp_activity_vector = torch.stack(cp_vectors).mean(dim=0)
                        else:
                            cp_activity_vector = torch.zeros(self.CP_FEATURE_DIM, dtype=torch.float32)
                    else:
                        cp_activity_vector = torch.zeros(self.CP_FEATURE_DIM, dtype=torch.float32)
                    if cp_activity_vector.shape[0] != self.CP_FEATURE_DIM:
                        correct_vector = torch.zeros(self.CP_FEATURE_DIM, dtype=torch.float32)
                        copy_len = min(cp_activity_vector.shape[0], self.CP_FEATURE_DIM)
                        correct_vector[:copy_len] = cp_activity_vector[:copy_len]
                        cp_activity_vector = correct_vector
                except Exception as e:
                    cp_activity_vector = torch.zeros(self.CP_FEATURE_DIM, dtype=torch.float32)

                # Create daily token
                daily_token = torch.cat([stock_features, cp_activity_vector])
                if daily_token.shape[0] != self.DAILY_TOKEN_DIM:
                    correct_token = torch.zeros(self.DAILY_TOKEN_DIM, dtype=torch.float32)
                    copy_len = min(daily_token.shape[0], self.DAILY_TOKEN_DIM)
                    correct_token[:copy_len] = daily_token[:copy_len]
                    daily_token = correct_token

                daily_sequence.append(daily_token)

            # Convert to tensor
            if daily_sequence:
                daily_sequence_tensor = torch.stack(daily_sequence)
            else:
                daily_sequence_tensor = torch.zeros((0, self.DAILY_TOKEN_DIM), dtype=torch.float32)

        # Get label
        label_key = f'Label_{self.label_period}'
        label = float(transaction_data['labels'].get(label_key, 0))

        return {
            'target_features': target_features,
            'daily_sequence': daily_sequence_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'transaction_id': tid
        }


def custom_collate_fn(batch):
    """Custom collate function"""
    target_features = torch.stack([item['target_features'] for item in batch])
    daily_sequences = [item['daily_sequence'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    transaction_ids = [item['transaction_id'] for item in batch]

    return {
        'target_features': target_features,
        'daily_sequences': daily_sequences,
        'label': labels,
        'transaction_id': transaction_ids
    }


def get_walk_forward_dataloaders(dataset_path, label_period='1M', batch_size=32, walk_forward_split=None,
                                 ablation_mode='full'):
    """Create dataloaders for a specific walk-forward split with ablation mode"""
    if walk_forward_split is None:
        raise ValueError("walk_forward_split must be provided")

    # Create datasets with ablation mode
    train_dataset = EfficientPerDayTradingDataset(
        dataset_path, label_period,
        transaction_ids=walk_forward_split['train_ids'],
        ablation_mode=ablation_mode
    )

    val_dataset = EfficientPerDayTradingDataset(
        dataset_path, label_period,
        transaction_ids=walk_forward_split['val_ids'],
        reference_scaler=train_dataset.trade_size_scaler,
        ablation_mode=ablation_mode
    )

    test_dataset = EfficientPerDayTradingDataset(
        dataset_path, label_period,
        transaction_ids = walk_forward_split['test_ids'],
    reference_scaler = train_dataset.trade_size_scaler,
    ablation_mode = ablation_mode
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    return train_loader, val_loader, test_loader
