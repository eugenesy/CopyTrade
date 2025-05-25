# Walk-Forward Copy-Trading Prediction

## What This Does

This system implements **walk-forward validation** for copy-trading prediction. It trains models on historical congressional trading data and tests on future periods, generating predictions with probabilities for each transaction.

**Walk-Forward Schedule:**
- Test 2020: Train on 2014-2019
- Test 2021: Train on 2015-2020
- Test 2022: Train on 2016-2021
- Test 2023: Train on 2017-2022
- Test 2024: Train on 2018-2023
- Test 2025: Train on 2019-2024

## Required Files

**You need these files in your folder:**
- `efficient_per_day_trading_dataset.pkl` - **Main dataset file (required!)**
- `build_efficient_per_day_dataset.py` - Builds the dataset
- `efficient_per_day_data_loader.py` - Data loading and walk-forward splits
- `model_components.py` - Neural network components
- `per_day_copy_trading_model.py` - Main model architecture
- `train_per_day.py` - Training script (this is what you run)

## File Descriptions

### **`build_efficient_per_day_dataset.py`**
Builds the main dataset from raw CSV files. Combines transaction data with stock history and creates the efficient sparse format.

### **`efficient_per_day_data_loader.py`**
- Creates walk-forward temporal splits based on filing dates
- Loads data with proper train/val/test splits for each time period
- Handles data preprocessing and batching

### **`model_components.py`**
Contains all neural network building blocks:
- CP embedding layers
- Transformer encoder
- Fusion modules
- Prediction head

### **`per_day_copy_trading_model.py`**
Main model architecture that combines all components into the final prediction model.

### **`train_per_day.py`**
**Main training script** - runs the complete walk-forward validation:
- Trains separate models for each test year
- Generates prediction CSVs with probabilities
- Creates performance summary across all years

## How to Run

### **1. Install Dependencies**
```bash
conda env create -f environment.yaml
conda activate copy-trading
```

### **2. Make Sure You Have the Dataset**
**Important:** You need `efficient_per_day_trading_dataset.pkl` in your folder. If you don't have it:
```bash
python build_efficient_per_day_dataset.py
```

### **3. Run Walk-Forward Validation**

**Basic usage:**
```bash
python train_per_day.py --dataset_path efficient_per_day_trading_dataset.pkl --label_period 1M
```

**For different time horizons:**
```bash
# 1-month predictions
python train_per_day.py --label_period 1M --output_dir results_1M

# 3-month predictions
python train_per_day.py --label_period 3M --output_dir results_3M

# 6-month predictions
python train_per_day.py --label_period 6M --output_dir results_6M
```

## Output Files

**Prediction files for each test year:**
- `predictions_2020_1M.csv` - Contains transaction_id, true_label, predicted_probability, predicted_label
- `predictions_2021_1M.csv`
- `predictions_2022_1M.csv`
- etc.

**Summary file:**
- `walk_forward_summary_1M.csv` - Performance metrics across all test years

## Requirements Files

### **`environment.yaml`**
```yaml
name: copy-trading
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch=1.12.0
  - torchvision=0.13.0
  - cudatoolkit=11.6
  - numpy=1.21.0
  - pandas=1.4.3
  - scikit-learn=1.1.1
  - matplotlib=3.5.2
  - seaborn=0.11.2
  - tqdm=4.64.0
  - python-dateutil=2.8.2
  - pip=22.1.2
  - pip:
    - transformers==4.21.0
    - tensorboard==2.9.1
```

### **`requirements.txt`**
```txt
torch==1.12.0
torchvision==0.13.0
numpy==1.21.0
pandas==1.4.3
scikit-learn==1.1.1
matplotlib==3.5.2
seaborn==0.11.2
tqdm==4.64.0
python-dateutil==2.8.2
transformers==4.21.0
tensorboard==2.9.1
```

## Command Line Options

**Key parameters:**
- `--label_period`: 1M, 3M, or 6M (prediction horizon)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Max epochs per model (default: 30)
- `--learning_rate`: Learning rate (default: 0.001)
- `--output_dir`: Where to save results

**Example with custom settings:**
```bash
python train_per_day.py \
    --label_period 1M \
    --batch_size 64 \
    --epochs 50 \
    --learning_rate 0.0005 \
    --output_dir custom_results
```

## Quick Checklist

Before running, make sure you have:
- [ ] `efficient_per_day_trading_dataset.pkl` in your folder
- [ ] All Python files in the same directory
- [ ] Dependencies installed
- [ ] Enough disk space for results

That's it! The system will automatically create temporal splits, train models for each year, and generate prediction files with probabilities.