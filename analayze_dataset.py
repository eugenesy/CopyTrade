import pickle
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_efficient_per_day_dataset(dataset_path="efficient_per_day_trading_dataset.pkl"):
    """Comprehensive analysis of the efficient per-day trading dataset"""

    print("=" * 80)
    print("EFFICIENT PER-DAY COPY-TRADING DATASET ANALYSIS")
    print("=" * 80)

    # Load dataset
    print("Loading dataset...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # Extract embeddings and main data
    embeddings_dict = dataset.pop('__embeddings__', {})
    transaction_ids = list(dataset.keys())

    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Total transactions: {len(transaction_ids)}")
    print(f"🧠 Embeddings available: {len(embeddings_dict)}")
    print()

    # ========== BASIC STRUCTURE ANALYSIS ==========
    print("🏗️  DATASET STRUCTURE")
    print("-" * 50)

    sample_id = transaction_ids[0]
    sample = dataset[sample_id]

    print("📁 Main components per transaction:")
    for key in sample.keys():
        print(f"   • {key}")

    print(f"\n📝 Sample transaction structure:")
    print(f"   • current_transaction: {len(sample['current_transaction'])} fields")
    print(f"   • stock_history: {len(sample['stock_history'])} days")
    print(f"   • cp_activity: {len(sample['cp_activity'])} active days")
    print(f"   • labels: {len(sample['labels'])} time periods")
    print()

    # ========== TRANSACTION FEATURES ANALYSIS ==========
    print("💼 TRANSACTION FEATURES")
    print("-" * 50)

    all_features = set()
    for tid in transaction_ids[:100]:  # Sample first 100
        all_features.update(dataset[tid]['current_transaction'].keys())

    print(f"📊 Available transaction features ({len(all_features)}):")
    for feature in sorted(all_features):
        print(f"   • {feature}")

    # Transaction types analysis
    tx_types = []
    tickers = []
    trade_sizes = []

    for tid in transaction_ids[:1000]:  # Sample for analysis
        tx = dataset[tid]['current_transaction']
        tx_types.append(tx.get('Transaction', 'Unknown'))
        tickers.append(tx.get('Appropriate_Ticker', 'Unknown'))
        trade_sizes.append(tx.get('Trade_Size_USD', '0'))

    print(f"\n📈 Transaction Types Distribution:")
    tx_type_counts = Counter(tx_types)
    for tx_type, count in tx_type_counts.most_common():
        percentage = (count / len(tx_types)) * 100
        print(f"   • {tx_type}: {count} ({percentage:.1f}%)")

    print(f"\n🏢 Top 10 Most Traded Tickers:")
    ticker_counts = Counter(tickers)
    for ticker, count in ticker_counts.most_common(10):
        percentage = (count / len(tickers)) * 100
        print(f"   • {ticker}: {count} ({percentage:.1f}%)")

    # ========== EMBEDDINGS ANALYSIS ==========
    print(f"\n🧠 EMBEDDINGS ANALYSIS")
    print("-" * 50)

    if embeddings_dict:
        sample_embedding = list(embeddings_dict.values())[0]
        print(f"📐 Embedding dimension: {len(sample_embedding)}")
        print(f"📊 Total embeddings: {len(embeddings_dict)}")

        # Check embedding statistics
        all_embeddings = np.array(list(embeddings_dict.values()))
        print(f"📈 Embedding statistics:")
        print(f"   • Mean: {all_embeddings.mean():.4f}")
        print(f"   • Std: {all_embeddings.std():.4f}")
        print(f"   • Min: {all_embeddings.min():.4f}")
        print(f"   • Max: {all_embeddings.max():.4f}")

    # ========== STOCK HISTORY ANALYSIS ==========
    print(f"\n📈 STOCK HISTORY ANALYSIS")
    print("-" * 50)

    stock_lengths = []
    stock_features = set()

    for tid in transaction_ids[:500]:  # Sample for analysis
        stock_hist = dataset[tid]['stock_history']
        stock_lengths.append(len(stock_hist))
        if stock_hist:
            stock_features.update(stock_hist[0].keys())

    print(f"📊 Stock history statistics:")
    print(f"   • Average length: {np.mean(stock_lengths):.1f} days")
    print(f"   • Min length: {min(stock_lengths)} days")
    print(f"   • Max length: {max(stock_lengths)} days")
    print(f"   • Stock features: {sorted(stock_features)}")

    # ========== CP ACTIVITY ANALYSIS ==========
    print(f"\n🤝 CP ACTIVITY ANALYSIS")
    print("-" * 50)

    activity_stats = []
    total_active_days = 0
    total_possible_days = 0

    for tid in transaction_ids[:500]:  # Sample for analysis
        cp_activity = dataset[tid]['cp_activity']
        stock_days = len(dataset[tid]['stock_history'])
        active_days = len(cp_activity)

        activity_stats.append({
            'active_days': active_days,
            'total_days': stock_days,
            'activity_rate': active_days / stock_days if stock_days > 0 else 0
        })

        total_active_days += active_days
        total_possible_days += stock_days

    print(f"📊 CP Activity statistics:")
    avg_active = np.mean([s['active_days'] for s in activity_stats])
    avg_total = np.mean([s['total_days'] for s in activity_stats])
    avg_rate = np.mean([s['activity_rate'] for s in activity_stats])

    print(f"   • Average active days per transaction: {avg_active:.1f}")
    print(f"   • Average total days per transaction: {avg_total:.1f}")
    print(f"   • Average activity rate: {avg_rate:.1%}")
    print(f"   • Overall activity rate: {total_active_days / total_possible_days:.1%}")

    # ========== LABELS ANALYSIS ==========
    print(f"\n🎯 LABELS ANALYSIS")
    print("-" * 50)

    label_stats = defaultdict(list)

    for tid in transaction_ids:
        labels = dataset[tid]['labels']
        for period, value in labels.items():
            if value is not None and not pd.isna(value):
                label_stats[period].append(value)

    print(f"📊 Label availability and distribution:")
    for period in ['Label_1M', 'Label_3M', 'Label_6M']:
        values = label_stats[period]
        if values:
            positive_rate = np.mean(values)
            print(f"   • {period}:")
            print(f"     - Available samples: {len(values)}")
            print(f"     - Positive rate: {positive_rate:.1%}")
            print(f"     - Class balance: {positive_rate:.3f} / {1 - positive_rate:.3f}")

    # ========== MEMORY EFFICIENCY ANALYSIS ==========
    print(f"\n💾 MEMORY EFFICIENCY ANALYSIS")
    print("-" * 50)

    total_possible_entries = 0
    total_stored_entries = 0

    for tid in transaction_ids[:100]:  # Sample for memory analysis
        stock_days = len(dataset[tid]['stock_history'])
        active_days = len(dataset[tid]['cp_activity'])

        total_possible_entries += stock_days
        total_stored_entries += active_days

    memory_savings = ((total_possible_entries - total_stored_entries) / total_possible_entries) * 100
    print(f"📊 Sparse storage efficiency:")
    print(f"   • Possible CP entries: {total_possible_entries:,}")
    print(f"   • Actual stored entries: {total_stored_entries:,}")
    print(f"   • Memory savings: {memory_savings:.1f}%")

    # ========== DATA QUALITY ANALYSIS ==========
    print(f"\n🔍 DATA QUALITY ANALYSIS")
    print("-" * 50)

    quality_issues = {
        'missing_embeddings': 0,
        'empty_stock_history': 0,
        'no_cp_activity': 0,
        'missing_labels': 0
    }

    for tid in transaction_ids[:1000]:  # Sample for quality check
        if tid not in embeddings_dict:
            quality_issues['missing_embeddings'] += 1

        if not dataset[tid]['stock_history']:
            quality_issues['empty_stock_history'] += 1

        if not dataset[tid]['cp_activity']:
            quality_issues['no_cp_activity'] += 1

        labels = dataset[tid]['labels']
        if all(v is None or pd.isna(v) for v in labels.values()):
            quality_issues['missing_labels'] += 1

    print(f"📊 Data quality metrics (sample of 1000):")
    for issue, count in quality_issues.items():
        percentage = (count / 1000) * 100
        print(f"   • {issue.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

    # ========== TEMPORAL ANALYSIS ==========
    print(f"\n📅 TEMPORAL ANALYSIS")
    print("-" * 50)

    trade_dates = []
    file_dates = []

    for tid in transaction_ids[:1000]:  # Sample for temporal analysis
        tx = dataset[tid]['current_transaction']
        trade_date = tx.get('Traded')
        file_date = tx.get('Filed')

        if trade_date:
            trade_dates.append(pd.to_datetime(trade_date))
        if file_date:
            file_dates.append(pd.to_datetime(file_date))

    if trade_dates:
        print(f"📊 Trading date range:")
        print(f"   • Earliest trade: {min(trade_dates).strftime('%Y-%m-%d')}")
        print(f"   • Latest trade: {max(trade_dates).strftime('%Y-%m-%d')}")
        print(f"   • Date span: {(max(trade_dates) - min(trade_dates)).days} days")

    if file_dates:
        print(f"📊 Filing date range:")
        print(f"   • Earliest filing: {min(file_dates).strftime('%Y-%m-%d')}")
        print(f"   • Latest filing: {max(file_dates).strftime('%Y-%m-%d')}")

    print(f"\n✅ Analysis completed!")
    print("=" * 80)

    return dataset, embeddings_dict


if __name__ == "__main__":
    dataset, embeddings = analyze_efficient_per_day_dataset()
