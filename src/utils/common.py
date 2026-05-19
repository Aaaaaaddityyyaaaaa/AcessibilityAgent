import yaml
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path

@ensure_annotations
def read_yaml(path : Path )->ConfigBox :
  with open(path , "r") as f : 
    content = yaml.safe_load(f) 
  return ConfigBox(content)