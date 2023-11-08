
from cProfile import label
from typing import Literal
from .dataset import Dataset
import pandas as pd
import numpy as np

class IrisDataset(Dataset):
  """Iris dataset class"""
  def __init__(self, path: str, mode: Literal['train', 'val', 'test'], split_by=[0.8, 0.1, 0.1], loggable=False, download=None):
    if download:
      super().from_download(path, download, loggable)
    else:
      super().__init__(path, loggable)

    self.classes = ["Setosa", "Versicolor", "Virginica"]
    self.idx_classes = [0, 1, 2]
    self.split_by = split_by
    self.features, self.labels, self.feature_names = self._make_dataset(path, mode)

  def split_dataset(self, df):
    return np.split(df.sample(frac=1, random_state=0), 
                                [int(self.split_by[0] * len(df)), int((self.split_by[0] + self.split_by[1]) * len(df))])

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

    data = []
    labels = []


    for feature in feature_names:
      if feature == 'variety':
        for label in df[feature].tolist():
          labels.append(0 if label == self.classes[0] else 1 if label == self.classes[1] else 2)
      else:  
        data.append((feature, df[feature].tolist()))
    return [x[1] for x in data], labels, [x[0] for x in data]
  

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
    return len(self.labels)
