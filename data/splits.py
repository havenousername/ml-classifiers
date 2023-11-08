from cgi import test
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

def get_dataset_partitions(df, train_split=0.8, val_split=0.1, test_split=0.1, target_variable=None):
  assert (train_split + test_split + val_split) == 1

  assert val_split == test_split

  # Shuffle
  df_sample = df.sample(frac=1, random_state=12)

  if target_variable is not None:
    grouped_df = df_sample.groupby(target_variable)
    arr_list = [np.split(g, [int(train_split * len(g)), int((1 - val_split) * len(g))]) for i, g in grouped_df]

    train_ds = pd.concat([t[0] for t in arr_list])
    val_ds = pd.concat([t[1] for t in arr_list])
    test_ds = pd.concat([t[2] for t in arr_list])
  else:
    indices_or_sections = [int(train_split * len(df)), int((1 - val_split) * len(df))]
    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
  return train_ds, val_ds, test_ds

