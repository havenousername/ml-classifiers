from abc import ABC, abstractmethod

from .downloader import Downloader # type: ignore

"""Dataset base class"""
class Dataset(ABC):
  """Abstact dataset base class"""
  def __init__(self, path: str, loggable = False):
    self.folder_path = path
    self.loggable = loggable
  
  @classmethod
  def from_download(self, path: str, url: str, loggable = False):
    name = url[url.rfind('/') + 1]
    downloader = Downloader(url, path, name, loggable)
    downloader.download_url()
    return self(path)

  @abstractmethod
  def __getitem__(self, index):
    """Return the sample at a given index"""
  
  @abstractmethod
  def __len__(self):
    """Return the size of dataset"""
  
  
