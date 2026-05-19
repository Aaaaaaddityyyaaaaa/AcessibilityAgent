import torch.nn as nn 
from transformers import GPT2Tokenizer , GPT2LMHeadModel
import joblib
from src.constants import CONFIG_PATH
from src.utils.common import read_yaml
import torch

class Model(nn.Module) :
  def __init__(self , path = CONFIG_PATH):
    super().__init__()
    self.config = read_yaml(path)
    self.projection = nn.Sequential(nn.Linear(2560,768),nn.ReLU(),nn.Dropout(0.1))
    self.gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    self.tokenizer.pad_token = self.tokenizer.eos_token
  
  def forward(self,features,captions) :
    
    projected = self.projection(features)
    prefix = projected.unsqueeze(1)
     
    if captions is not None :
      tokens = self.tokenizer(captions ,return_tensors="pt",padding=True,truncation=True,max_length=128).input_ids.to(projected.device)
      embeddings = self.gpt_model.transformer.wte(tokens)
      inputs = torch.cat([prefix , embeddings],dim=1)
      labels = tokens.clone()
      labels = torch.cat([
            torch.full((labels.size(0), 1), -100,device=labels.device),
            labels
        ], dim=1)
      outputs = self.gpt_model(
            inputs_embeds=inputs,
            labels=labels
        )
      return outputs.loss
    else :
      caption = self.inference(feature=features)
      return caption
  
  def inference(self , feature) :
    with torch.no_grad() :

      projected = self.projection(feature)
      prefix = projected.unsqueeze(1)
      caption = self.gpt_model.generate(inputs_embeds=prefix,
                max_length=50,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id)
      return self.tokenizer.decode(             
                caption[0],
                skip_special_tokens=True
            )


      
    
