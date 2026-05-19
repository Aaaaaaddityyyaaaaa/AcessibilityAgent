from langchain_groq import ChatGroq
from dotenv import load_dotenv
from os import getenv
from src.utils.common import read_yaml
from pathlib import Path 
from src.constants import CONFIG_PATH



load_dotenv()
def get_llm(path = CONFIG_PATH) :
  config = read_yaml(path)
  llm = ChatGroq(model= config.llm.name,api_key=getenv("GROQ_API_KEY") , temperature=config.llm.temperature)
  return llm