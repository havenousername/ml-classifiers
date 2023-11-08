from typing import Literal
from typing import Type
import math
import sys 
from data.dataloader import DataLoader

"""
K-NN algorithm
Steps to solve
-Define the distance measure
-Compute the nearest neighbour
-Classify 
"""
class KNNClassifier:
  @staticmethod
  def eucledean(p: list, q: list):
    sum = 0
    for index in range(len(p)):
      sum += (p[index] - q[index])**2
    return math.sqrt(sum)
  
  @staticmethod
  def l1(p: list, q: list):
    sum = 0
    for index in range(len(p)):
      sum += abs((p[index] - q[index]))
    return sum
    

  def __init__(self, K: int, distance_type: Literal['eucledean'], x_train: list | Type[DataLoader] , x_data: list | Type[DataLoader], labels: list | None, classes = [0, 1]) -> None:
    self.K = K 
    
    if (distance_type == 'eucledean'):
      self.distance = KNNClassifier.eucledean
    elif (distance_type == 'l1' or distance_type == 'manhattan'):
      self.distance = KNNClassifier.l1
    else:
      raise ValueError(f"{distance_type} is not yet supported. Try eucleadean distance instead") 
    
    self.classes = classes

    if not isinstance(x_data, list):
      features, labels = self._make_data(x_data)
      self.x_data = features
      self.data_labels = labels
    else:
      self.x_data = x_data
      self.data_labels = []
    
    if not isinstance(x_train, list):
      features, labels = self._make_data(x_train)
      self.labels = labels
      self.x_train = features
    else:
      self.labels = labels
      self.x_train = x_train 
    

  def _make_data(self, data):
    features = []
    labels = []
    for x in data:
      features.append(x['features'])
      labels.append(x['label'])
    return features[0], labels[0]

  def eval(self):
    predicted = []
    for data in self.x_data:
      data_probs = [(y, self._label_probability(y, data)) for y in self.classes]
      data_probs = sorted(data_probs, key=lambda x: x[1], reverse=True)
      predicted.append(data_probs[0])
    return [y[0] for y in predicted]
  
  def change_data(self, x_data):
    if not isinstance(x_data, list):
      features, labels = self._make_data(x_data)
      self.x_data = features
      self.data_labels = labels
    else:
      self.x_data = x_data
      self.data_labels = []
  
  def _label_probability(self, label: int, x: list):
    # distance of current x to the other points 
    distances = [(self.distance(data_point, x), self.labels[index]) 
                 for index, data_point in enumerate(self.x_train)]
  
    distances = sorted(distances, key= lambda x: x[0])[:self.K]
    # neighbours of y
    neightbours = [x[1] for x in distances]
    total = 0
    for y in neightbours:
      total += self._Ie(y == label) 
    # print(neightbours, total, label)
    return (1/self.K) * total 

  def _Ie(self, b: bool):
    if b: return 1
    else: return 0


"""
K-NN with weights algorithm
Descendant from KNN class
"""
class KNNWeightedClassifier(KNNClassifier):
  def __init__(self, K: int, distance_type: Literal['eucledean'], x_train: list | type[DataLoader], x_data: list | type[DataLoader], labels: list | None, classes=[0,1]) -> None:
    super().__init__(K, distance_type, x_train, x_data, labels, classes)

  
  def _label_probability(self, label: int, x: list):
    # distance of current x to the other points 
    distances = [(self.distance(data_point, x), self.labels[index]) 
                 for index, data_point in enumerate(self.x_train)]
  
    distances = sorted(distances, key= lambda x: x[0])[:self.K]
    z = 0
    total = 0
    for y in distances:
      distance = sys.maxsize if y[0] == 0 else 1/y[0]
      z += distance
      total += distance * self._Ie(y[1] == label) 
    return (1/z) * total 