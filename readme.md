This repository provides end-to-end tools to build an efficient per-day copy-trading dataset and train a Transformer-based prediction model using a 1-month return label. Follow the sections below to install dependencies, prepare data, and run training.

## Repository Structure  
- **data/**  
  - `transactions_with_embeddings_enhanced.csv` – Raw transactions enriched with sentence embeddings  
  - `all_tickers_historical_data.pkl` – Pickled historical stock data  
- **scripts/**  
  - `build_efficient_per_day_dataset_enhanced.py` – Build sparse per-day CP activity dataset  
  - `efficient_per_day_data_loader.py` – PyTorch Dataset, DataLoader and walk-forward split utilities  
  - `model_components.py` – Core model building blocks (CP embedding, attention, transformer, fusion, head)  
  - `per_day_copy_trading_model.py` – Full model definition with ablation modes  
  - `train_per_day.py` – Training, validation, testing and summary generation script  
- **outputs/** (generated)  
  - `walk_forward_results/` – Model metrics CSVs and prediction files  
  - `efficient_per_day_trading_dataset_enhanced.pkl` – Final dataset pickle  

## Prerequisites  
- Python 3.8+  
- Required packages (install via pip):  
  ```bash
  pip install pandas numpy torch sklearn tqdm python-dateutil matplotlib seaborn
  ```

## Data Preparation  
1. Place your raw CSV and stock history pickle under `data/`.  
2. Run the dataset builder to produce `efficient_per_day_trading_dataset_enhanced.pkl`:  
   ```bash
   python scripts/build_efficient_per_day_dataset_enhanced.py \
     --transactions_csv data/transactions_with_embeddings_enhanced.csv \
     --stock_history_pkl data/all_tickers_historical_data.pkl \
     --output_pkl outputs/efficient_per_day_trading_dataset_enhanced.pkl
   ```
   This script:  
   - Extracts the last 90 trading days of stock features (pads if needed)  
   - Builds a sparse mapping of CP trades per day  
   - Saves periodic checkpoints for fault tolerance  

## Training & Evaluation  
All training, validation and testing use a walk-forward scheme starting in 2020 with a rolling 6-year training window. Labels are 1-month future returns only.

1. **Run training**  
   ```bash
   python scripts/train_per_day.py \
     --dataset_path outputs/efficient_per_day_trading_dataset_enhanced.pkl \
     --label_period 1M \
     --batch_size 32 \
     --learning_rate 0.001 \
     --epochs 30 \
     --output_dir outputs/walk_forward_results \
     --ablation_mode full
   ```
   Key arguments:  
   - `--label_period 1M` (fixed)  
   - `--ablation_mode` choices:  
     - `full` (stock + target CP)  
     - `target_cp_only` (CP features only)  
     - `historical_only` (historical data only)  

2. **Monitor progress**  
   - Training/validation loss and AUC printed per epoch  
   - Early stopping after 7 epochs of no improvement  

3. **Outputs generated**  
   - `walk_forward_summary_1M.csv` – Year-by-year metrics (AUC, F1, accuracy)  
   - `predictions__1M.csv` – Transaction-level true label, probability and predicted label  

## Notes & Tips  
- Adjust hyperparameters (e.g. transformer layers, fusion dim) via CLI flags in `train_per_day.py`.  
- To reproduce results, fix random seeds in your Python environment.  
- For large datasets, ensure enough disk space for checkpoints in `outputs/`.  

This setup enables rapid experimentation on 1-month return prediction using per-day market and copy-trading signals.

## XGBoost
The XGBoost_Baseline.ipynb includes the training and evaluation of XGboost model.
## Dataset
The dataset for csv version in provide in the URL due to github limitation
https://drive.google.com/drive/folders/12xj-P2yK5kkX5KV4oDQGUCKnxjfgongF?usp=sharing
## Backtrade model 
The  Backtrade model is provided in Portfolio_Allocation(2020_2025).ipynb


