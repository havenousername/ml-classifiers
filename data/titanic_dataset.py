"""
Definition of the Titanic dataset class
"""
from typing import Literal

from .dataset import Dataset 
import pandas as pd # type: ignore
import numpy as np # type: ignore
from splits import get_dataset_partitions

class TitanicDataset(Dataset):
  """Titanic dataset class"""
  def __init__(self, path: str, mode: Literal['train', 'val', 'test'], split_by=[0.8, 0.1, 0.1], loggable=False):
    super().__init__(path, loggable)
    self.classes = ["died", "survided"]
    self.idx_classes = [0, 1]
    self.split_by = split_by
    self.features, self.labels, self.feature_names, self.ids = self._make_dataset(path, mode)
  
  def split_dataset(self, df):
    if self.split_by[0] == 1:
      return df, [], []
    return get_dataset_partitions(df, self.split_by[0],self.split_by[1], self.split_by[2], 0)

  def _make_dataset(self, path: str, mode: Literal['train', 'val', 'test']):
    dataset = pd.read_csv(path)
    df = dataset.dropna()

    train, val, test = self.split_dataset(df)
    

    if mode == 'train':
      df = train 
    elif mode == 'val':
      df = val
    elif mode == 'test':
      df = test

    
    feature_names = df.columns

    # arrays of dataset
    labels = []
    # data matrix of the features
    # N(features)xM(feature_entries)
    data = []
    ids = []

    for feature in feature_names:
      if feature == 'Survived':
        for label in df[feature].tolist():
          labels.append(label)
      elif feature == 'PassengerId':
        for id in df[feature].tolist():
            ids.append(id)
      elif feature == 'Name':
        continue
      elif feature == 'Ticket' or feature == 'Cabin':
        continue 
      elif feature == 'Sex':
        data.append((feature, [1 if x == 'female' else 0 for x in df[feature].tolist()]))
      elif feature == 'Embarked':
        continue
        # data.append((feature, [2 if x == 'C' else 1 if x == 'Q' else 0 for x in df[feature].tolist()]))
      else:
        data.append((feature, df[feature].tolist()))
    return  [x[1] for x in data], labels, [x[0] for x in data], ids

  def __getitem__(self, index):
    """
    Get a sigle item from which is dictionary
    """
    return { 
      "features": list(map(lambda feature: feature[index], self.features)),   
      "label": self.labels[index],
      "feature_names": self.feature_names,
    }
  
  def __len__(self):
    return len(self.features)

