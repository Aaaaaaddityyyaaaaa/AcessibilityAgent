import torchvision.models as models
import torch.nn as nn 
from ..constants import CONFIG_PATH
from ..utils.common import read_yaml
import joblib

def create_image_model(path = CONFIG_PATH) :
  model = models.efficientnet_b7(weights= models.EfficientNet_B7_Weights.IMAGENET1K_V1)
  model.classifier[1] = nn.Identity()
  config = read_yaml(path)
  model.eval()
  joblib.dump(model,config.feature_extractor.path)

if __name__ == "__main__" :
  create_image_model()
  