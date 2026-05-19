from torch.utils.data import Dataset
from .image_preprocess import preprocess
import os
import joblib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.common import read_yaml 
from src.constants import CONFIG_PATH
import pandas as pd
import torch

class Flickr8k(Dataset) :
  def __init__(self , path = CONFIG_PATH) :
    config = read_yaml(path)
    if os.path.exists(config.data.datasetPkl):
      self.data = pd.read_pickle(config.data.datasetPkl)
    else :
      model = joblib.load("model/feature_model/f_model.pkl")
      model.eval()
      self.data = pd.read_csv(config.data.captions , sep=",",names=["image", "caption"])
      count =  0 
      features = {}
      with torch.no_grad() :
        for img in os.listdir(config.data.images) :
          if(count>=1000) :
            break
          
          count +=1
          image = preprocess(img , path=CONFIG_PATH)
          feature = model(image)
          image_str = str(img)
          feature = feature.squeeze(0)
          features[image_str] = feature
      
      self.data = self.data[
              self.data["image"].isin(features.keys())         
          ].reset_index(drop=True)

      self.data["feature"] = self.data["image"].map(features)

      pd.to_pickle(self.data ,  config.data.datasetPkl)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    feature = torch.tensor(row["feature"])  
    caption = row["caption"]
    return feature, caption