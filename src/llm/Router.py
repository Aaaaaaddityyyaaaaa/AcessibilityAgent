from src.llm.state import State
def router(state:State) :
  if(state.get("image") is not None) :
    return "captioning"
  else :
    return "llm"

  