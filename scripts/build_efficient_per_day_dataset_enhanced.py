import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

# Configuration
TRANSACTIONS_CSV = '/home/syeugenel/FINTECH/Project 3/5. Sentence Embeddings/transactions_with_embeddings_enhanced.csv'
STOCK_HISTORY_PKL = '/home/syeugenel/FINTECH/Project 3/1. Download Historical Data/all_tickers_historical_data.pkl'
OUTPUT_PKL = 'efficient_per_day_trading_dataset_enhanced.pkl'
CHECKPOINT_FILE = 'efficient_per_day_dataset_checkpoint_enhanced.pkl'
CHECKPOINT_INTERVAL = 10000


def save_checkpoint(dataset, processed_ids):
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(dataset, f)
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump({'processed_ids': processed_ids}, f)
    print(f"Checkpoint saved: {len(processed_ids)} transactions processed")


def load_checkpoint():
    if os.path.exists(OUTPUT_PKL) and os.path.exists(CHECKPOINT_FILE):
        with open(OUTPUT_PKL, 'rb') as f:
            dataset = pickle.load(f)
        with open(CHECKPOINT_FILE, 'rb') as f:
            data = pickle.load(f)
        processed_ids = set(data['processed_ids'])
        print(f"Loaded checkpoint with {len(processed_ids)} processed transactions")
        return dataset, processed_ids
    return {}, set()


def build_sparse_cp_activity(df, ticker, stock_dates, filed_date):
    """Optimized: Build sparse per-day CP activity - only store dates with actual activity"""
    # Pre-filter dataframe once for this ticker and filed_date
    relevant_df = df[
        (df['Appropriate_Ticker'] == ticker) &
        (df['Filed'] < filed_date) &
        (df['Traded'].dt.strftime('%Y-%m-%d').isin(stock_dates))
    ].copy()

    if relevant_df.empty:
        return {}

    # Convert traded dates to string for mapping
    relevant_df['Traded_str'] = relevant_df['Traded'].dt.strftime('%Y-%m-%d')

    # Group by traded date and collect IDs
    grouped = relevant_df.groupby('Traded_str')['ID'].apply(list)

    return grouped.to_dict()


def extract_stock_history_with_dates(ticker, trade_date, file_date, stock_data):
    """Extract exactly last 90 stock data points before filing"""
    if ticker not in stock_data or not stock_data[ticker]:
        return [], []

    all_dates = sorted(stock_data[ticker])
    file_date_str = pd.to_datetime(file_date).strftime('%Y-%m-%d')

    # Get dates before filing
    relevant_dates = [d for d in all_dates if d < file_date_str]

    # Take last 90 points (or all if fewer than 90)
    last_90_dates = relevant_dates[-90:] if len(relevant_dates) >= 90 else relevant_dates

    # Pad if less than 90
    if len(last_90_dates) < 90:
        pad_count = 90 - len(last_90_dates)
        if last_90_dates:
            # Use earliest date data for padding
            earliest_data = stock_data[ticker][last_90_dates[0]]
            stock_history = [earliest_data.copy() for _ in range(pad_count)]
            for data in stock_history:
                data['date'] = 'PAD'
        else:
            stock_history = [{'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0, 'date': 'PAD'} for _ in
                             range(pad_count)]

        # Add actual data
        for date in last_90_dates:
            data = stock_data[ticker][date].copy()
            data['date'] = date
            stock_history.append(data)

        return stock_history, ['PAD'] * pad_count + last_90_dates
    else:
        stock_history = []
        for date in last_90_dates:
            data = stock_data[ticker][date].copy()
            data['date'] = date
            stock_history.append(data)
        return stock_history, last_90_dates


def main():
    print("Loading data sources...")
    df = pd.read_csv(TRANSACTIONS_CSV, parse_dates=['Traded', 'Filed'])

    if 'ID' not in df.columns:
        print("Error: ID column missing from transactions data")
        return

    with open(STOCK_HISTORY_PKL, 'rb') as f:
        stock_data = pickle.load(f)

    # Load checkpoint
    dataset, processed_ids = load_checkpoint()

    # Extract embeddings dictionary
    print("Extracting embeddings...")
    embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
    embeddings_dict = {}
    embeddings_df = df[['ID'] + embedding_cols].dropna()
    embeddings_dict = dict(zip(embeddings_df['ID'], embeddings_df[embedding_cols].values.tolist()))

    total_ids = len(df)

    print(f"Building efficient per-day dataset for {total_ids} transactions...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        transaction_id = row['ID']

        if transaction_id in processed_ids:
            continue

        if pd.isna(row['Appropriate_Ticker']) or pd.isna(row['Traded']) or pd.isna(row['Filed']):
            processed_ids.add(transaction_id)
            continue

        # Extract stock history with dates
        stock_history, stock_dates = extract_stock_history_with_dates(
            row['Appropriate_Ticker'],
            row['Traded'],
            row['Filed'],
            stock_data
        )

        # Build sparse CP activity (only store dates with activity)
        sparse_cp_activity = build_sparse_cp_activity(
            df,
            row['Appropriate_Ticker'],
            stock_dates,
            row['Filed']
        )

        # Create entry
        entry = {
            'current_transaction': {
                col: row[col] for col in row.index
                if col not in ['ID'] + [f'Label_{p}' for p in ['1M', '3M', '6M']] + embedding_cols
            },
            'stock_history': stock_history,
            'cp_activity': sparse_cp_activity,  # 🔥 NEW: Only dates with activity
            'labels': {
                'Label_1M': row.get('Label_1M'),
                'Label_3M': row.get('Label_3M'),
                'Label_6M': row.get('Label_6M')
            }
        }

        dataset[transaction_id] = entry
        processed_ids.add(transaction_id)

        if len(processed_ids) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(dataset, processed_ids)

    # Add embeddings dictionary
    dataset['__embeddings__'] = embeddings_dict

    # Final save
    save_checkpoint(dataset, processed_ids)
    print(f"Dataset complete with {len(dataset) - 1} transactions")

    # Create sample analysis
    print("\n=== MEMORY EFFICIENCY ANALYSIS ===")
    sample_id = list(dataset.keys())[0]
    if sample_id != '__embeddings__':
        sample_data = dataset[sample_id]
        stock_days = len(sample_data['stock_history'])
        activity_days = len(sample_data['cp_activity'])

        print(f"Sample transaction {sample_id}:")
        print(f"Stock history days: {stock_days}")
        print(f"Days with CP activity: {activity_days}")
        print(f"Memory savings: {((stock_days - activity_days) / stock_days * 100):.1f}% fewer stored entries")

        if sample_data['cp_activity']:
            print("\nDays with actual CP activity:")
            for date, cps in sample_data['cp_activity'].items():
                print(f"  {date}: {len(cps)} CPs traded")
        else:
            print("No CP activity found in this sample period")


if __name__ == "__main__":
    main()
