# src/llm/model_loader.py
import joblib
from src.utils.common import read_yaml
from src.constants import CONFIG_PATH
from src.captioninng_model.get_gpt2_decoder import Model
import torch
from data.image_preprocess import preprocess_64
from langchain_core.tools import tool
from src.llm.state import State

config = read_yaml(CONFIG_PATH)

f_model = joblib.load(config.feature_extractor.path)
decoder_model = Model(CONFIG_PATH)

decoder_weights = joblib.load(config.training.save_path)
decoder_model.projection.load_state_dict(decoder_weights["projection"])
decoder_model.gpt_model.load_state_dict(decoder_weights["gpt_model"])

f_model.eval()
decoder_model.eval()





def captioningAgent(state: State):

    """Generates a caption for an uploaded image and stores it in state"""
    
    img_tensor = preprocess_64(state.get("image", ""))
    with torch.no_grad():
        feature = f_model(img_tensor)
        caption = decoder_model.inference(feature)
    state["text"] = caption
    return state