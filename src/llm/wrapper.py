from src.llm.AgentGraph import create_graph 
class AgentWrapper :
  def __init__(self) :
    self.app = create_graph()
  def response(self , state) :
    answer  = self.app.invoke(state)
    return answer
    

