from typing import TypedDict , Optional
class State(TypedDict) :
  image : Optional[str]
  prompt : Optional[str] 
  text : Optional[str]
  audio : Optional[bytes]