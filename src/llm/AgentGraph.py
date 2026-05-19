from typing import TypedDict , Optional
from langgraph.graph import StateGraph ,START
from src.llm.Router import router
from src.llm.captioning_agent import captioningAgent
from src.llm.llm_agent import llmAgent
from src.llm.tts_agent import ttsAgent
from src.llm.state import State



def create_graph() :
  graph = StateGraph(State)
  
  graph.add_node("Captioning",captioningAgent)
  graph.add_node("LLM",llmAgent)
  graph.add_node("TTS" ,ttsAgent)
  graph.add_conditional_edges(START,path=router,path_map={"captioning": "Captioning",
        "llm": "LLM"})
  graph.add_edge("LLM","TTS")
  graph.add_edge("Captioning","TTS")
  
  graph.set_finish_point("TTS")
  app = graph.compile()
  return app