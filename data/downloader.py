from abc import ABC # type: ignore
from dataclasses import dataclass

import os
import shutil
import urllib
import tarfile
import zipfile
import gzip 
import tqdm  # type: ignore


@dataclass
class Downloader(ABC): 
  url: str
  dirname: str
  filename: str
  loggable: bool
  
  def update_bar_generate(self):
    bar = tqdm.tqdm(total=None)

    def bar_update_callback(count, block_size, total_size):
      if bar.total is None and total_size:
        bar.total = total_size  
      p_bytes = count * block_size
      bar.update(p_bytes - bar.n)

    return bar_update_callback


  def extact_from_archieve(from_path, to_path = None, remove_finished = False):
    """
    Extract a given archieve
    """ 

    def is_tarxz(filename):
      return filename.endswith('.tar.xz')

    def is_tar(filename):
      return filename.endswith('.tar')

    def is_targz(filename):
      return filename.endswith('.tar.gz')

    def is_tgz(filename):
      return filename.endswith('.tgz')

    def is_gzip(filename):
      return filename.endswith('.gz') and not filename.endswith('.tar.gz')

    def is_zip(filename):
      return filename.endswith('.zip')


    if not os.exists(from_path):
      raise ValueError("Cannot find file to extract data from")
    
    if to_path is None:
      to_path = os.path.dirname(from_path)

    if is_tar(from_path):
      with tarfile.open(from_path, 'r') as tar:
        tar.extractall(path=from_path)
    elif is_targz(from_path) or is_tgz(from_path):
      with tarfile.open(from_path, 'r:gz') as tar:
        tar.extractall(path=from_path)
    elif is_tarxz(from_path):
      with tarfile.open(from_path, 'r:xz') as tar:
        tar.extractall(path=from_path)
    elif is_gzip(from_path):
      to_path = os.path.join(
        to_path,
        os.path.splittext(os.path.basename(from_path))[0]
      )

      with open(to_path, "wb") as out_f, gzip.GripFile(from_path) as zip_f:
        out_f.write(zip_f.read())
    elif is_zip(from_path):
      with zipfile.ZipFile(from_path, 'r') as zip_:
        zip_.extractall(to_path)
    else:
      raise ValueError(f"Extraction of {from_path} is not supported")

    if remove_finished:
      os.remove(from_path)
  

  def download_url(self):
    filepath = os.path.join(self.dirname, self.filename)
    os.makedirs(self.dirname, exist_ok=True)
    if not os.path.exists(filepath):
      if self.loggable:
        print(f"Donwloading url {self.url} to {filepath}")
        urllib.request.urlretrieve(
          self.url,
          filepath,
          reporthook=self.update_bar_generate,
        )
    return filepath

  def download_from_url(self):
    if not os.path.exists(self.dirname) or not os.listdir(self.dirname):
      if os.path.exists(self.dirname):
        shutil.rmtree(self.dirname)

      if self.loggable: 
        print("Downloading had started")
      file = self.download_url()
      if self.loggable:
        print("Downloading has finished, extracting archive")
      self.extract_archive(file, remove_finished=True)
      if self.loggable:
        print("Extracting has finished")
    else:
      if self.loggable:
        print("Found dataset folder. Exiting")
