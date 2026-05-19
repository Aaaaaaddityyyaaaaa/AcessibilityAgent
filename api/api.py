from fastapi import FastAPI
from pydantic import BaseModel
from src.llm.wrapper import AgentWrapper
import json
from typing import Optional
import base64

class RequestBody(BaseModel):
  image : Optional[str] =None
  prompt : str =""

app = FastAPI()
agent_app = AgentWrapper()
@app.post("/Agent")
def makeCall(body:RequestBody) :
  state={
    "image":body.image ,
    "prompt" : body.prompt  , 
    "text"  : "" , 
    "audio" : None 
  }
  response = agent_app.response(state)
  if response.get("audio"):
    response["audio"] = base64.b64encode(response["audio"]).decode()

  return response
  
  
