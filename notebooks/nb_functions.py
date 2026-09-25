from glob import glob
import pandas as pd

# Returns all the processed datasets needed for model training
def read_proc_data():
  proc_datasets = glob(r'..\data\processed\*.parquet')
  target_col = 'RainTomorrow'

  for path in proc_datasets:
    if 'train_inputs' in path:
      train_inputs = pd.read_parquet(path)
    elif 'val_inputs' in path:
      val_inputs = pd.read_parquet(path)
    elif 'test_inputs' in path:
      test_inputs = pd.read_parquet(path)
    elif 'train_target' in path:
      train_target = pd.read_parquet(path)[target_col]
    elif 'val_target' in path:
      val_target = pd.read_parquet(path)[target_col]
    else:
      test_target = pd.read_parquet(path)[target_col]

  return train_inputs, val_inputs, test_inputs, train_target, val_target, test_target