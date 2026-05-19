import os 
from os import getenv
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  
import kagglehub.datasets
from src.utils.common import read_yaml
from dotenv import load_dotenv
import kagglehub
from src.constants import CONFIG_PATH


load_dotenv()

def get_data_set(path = CONFIG_PATH) :
  config = read_yaml(path)
  kagglehub.dataset_download(handle="adityajn105/flickr8k" , output_dir=  config.data.dataset_zipped , force_download=False )


if __name__ == "__main__" :
  get_data_set()