from langchain_core.tools import tool
from src.llm.llm import get_llm
from src.llm.state import State

def llmAgent(state:State) :
  """Called if answers are needed"""
  model  = get_llm() 
  answer = model.invoke(input=state.get("prompt",""))
  state["text"] = answer.content
  return state