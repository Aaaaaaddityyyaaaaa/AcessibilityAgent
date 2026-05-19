from langchain_core.tools import tool
import gtts
import io

from src.llm.state import State

def ttsAgent(state:State) :
  """It gets called when text needs to be converted into audio description"""
  tts = gtts.gTTS(state.get("text" , "") , lang="en")
  audio_buffer = io.BytesIO()
  tts.write_to_fp(audio_buffer)
  audio_buffer.seek(0)
  state["audio"] = audio_buffer.read()
  return  state