from PIL import Image
from torchvision import transforms
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.common import read_yaml 
from src.constants import CONFIG_PATH  
import pickle
import base64
import io
import base64

def preprocess(name , path = CONFIG_PATH ) :
  config = read_yaml(path)
  image = Image.open(f"{config.data.images}/{name}").convert("RGB")
  transform = transforms.Compose([
    transforms.Resize((600,600)),          # EfficientNet expects 224x224
    transforms.ToTensor(),                  # PIL image → tensor (C, H, W), values 0-1
    transforms.Normalize(                   # normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],         # same stats EfficientNet was trained on
        std=[0.229, 0.224, 0.225]
    )
])
  image = transform(image)
  image = image.unsqueeze(0)
  return image

def preprocess_64(image_base64:str )  :
  
  image_bytes = base64.b64decode(image_base64)
  image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

  transform = transforms.Compose([
    transforms.Resize((600,600)),          # EfficientNet expects 224x224
    transforms.ToTensor(),                  # PIL image → tensor (C, H, W), values 0-1
    transforms.Normalize(                   # normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],         # same stats EfficientNet was trained on
        std=[0.229, 0.224, 0.225]
    )
])
  image = transform(image)
  image = image.unsqueeze(0)
  return image
