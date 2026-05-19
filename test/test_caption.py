import base64
from src.llm.captioning_agent import captioningAgent
from src.constants import CONFIG_PATH
with open("data/dataset/zipped/images/10815824_2997e03d76.jpg","rb") as f :
  image_64 = base64.b64encode(f.read()) 


print(captioningAgent.func(image_64,CONFIG_PATH))