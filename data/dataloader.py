
from dataclasses import dataclass
from locale import normalize
from typing import Type

from .dataset import Dataset
import math
import sys

import numpy as np # type: ignore

@dataclass
class DataLoader:
  """
  Dataloader Class
  Defines an iterable batch-sampler over a given dataset
  """ 
  dataset: Type[Dataset]
  batch_size: int = 0
  shuffle: bool = False
  normalization: bool = True

  def __post_init__(self):
    if self.batch_size == 0:
      self.batch_size = len(self.dataset)

  
  def normalize(self):
    mins = self.dataset[0]["features"]
    maxs = self.dataset[-1]["features"]
    for data in self.dataset: 
      mins = [data["features"][idx] if data["features"][idx] < x else x  for idx, x in enumerate(mins)]
      maxs = [data["features"][idx] if data["features"][idx] > x else x  for idx, x in enumerate(maxs)]

    # print(maxs, mins)
    # print(self.dataset[0]["features"], maxs, mins)
    dataset = []

    def local_log(x):
      if x == 0:
        return 0
      return math.log(x)
    
    for data in self.dataset:
        dataset.append({
          "features": [
            local_log(x - mins[idx] / (maxs[idx]-mins[idx])) if data["feature_names"][idx] == 'Fare' else 
                       (x - mins[idx]) / (maxs[idx]-mins[idx])  for idx, x in enumerate(data["features"])],
          "label":  data["label"],
          "feature_names": data["feature_names"]
        })
    return dataset
  
  def __iter__(self):
    def combine_batch_dicts(batch):
      batch_dict = {}
      for data_dict in batch:
        for key,value in data_dict.items():
          if key not in batch_dict:
            batch_dict[key] = []
          batch_dict[key].append(value) 
      return batch_dict
    
    dataset = self.normalize() if self.normalization else self.dataset
    if self.shuffle:
      index_iterator = iter(np.random.permutation(len(dataset)))
    else:
      index_iterator = iter(range(len(dataset)))

    batch = []    
    
    for index in index_iterator:
      batch.append(dataset[index])
    if len(batch) > 0:
      yield combine_batch_dicts(batch)

  def __len__(self):
    return int(np.ceil(len(self.dataset) / self.batch_size))