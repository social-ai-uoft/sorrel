from sorrel.models import BaseModel
from sorrel.buffers import StrBuffer
import openai

class Client:
  def __init__(
      self, 
      model: str,
      max_tokens: int = 4096
  ): 
    
    self.model = model
    self.client = openai.OpenAI(
      api_key="ollama",
      base_url="http://localhost:11434/v1/",
    )
    self.msg_history = []
    self.max_tokens = max_tokens

  def clear(self):
    """Clear message history."""
    self.msg_history = []

  def call(
      self,
      prompt: str,
      system_message: str,
      temperature: float = 0.7,
  ):
    self.msg_history += [{
      "role": "user",
      "content": prompt
    }]
    response = self.client.chat.completions.create(
      model=self.model,
      messages=[
        {"role": "system", "content": system_message},
        *self.msg_history
      ],
      temperature=temperature,
      max_tokens=self.max_tokens,
      n=1,
      stop=None
    )
    content = response.choices[0].message.content
    self.msg_history += [{
      "role": "assistant",
      "content": content
    }]
    return content

class LLM(BaseModel):
  """Interface for large language model."""
  def __init__(
      self,
      action_space,
      memory_size,
      action_list: list,
      model_name: str,
      instructions: str,
      temperature: float = 0.7,
      max_tokens: int = 4096,
      **call_params
  ):
    super().__init__(1, action_space, memory_size)

    self.action_list = action_list
    self.call_params = call_params
    self.model = model_name
    self.instructions = instructions
    self.client = Client(
      model_name,
      max_tokens=max_tokens
    )
    self.stm = ""
    self.memory = StrBuffer(
      capacity=memory_size,
      obs_shape=[11, 11]
    )
    self.temperature = temperature

  def take_action(self, state) -> int:

    output = self.client.call(
      prompt=state,
      system_message=self.instructions,
      temperature=self.temperature
    )
    response = output.lower() #type: ignore
    return self.action_list.index(response)
  
  def format_memories(self, memories):
    
    states, actions, rewards, dones = memories
    memories = list(zip(states, actions, rewards, dones))
    memories = [f"Memory:\n=======\n{state}\nAction: {action}\nReward: {reward}\nDone: {done}\n" for state, action, reward, done in memories]
    print(memories[0])

    return memories

  def recall(self, k: int = 1, method: str = 'recency'):
    """Recall the `k` most recently observed states and store 
    them in the short-term memory buffer.
    
    Args:
      k (int): The number of memories to recall.
      method (str): The method for recall. By default, recency.
    """
    if method == "recency":
      memories = self.format_memories(self.memory[self.memory.idx-k:self.memory.idx])
      self.stm = "\n\n".join(memories)
    elif method == 'frequency':
      # TODO: Implement frequency-based recall
      pass
    else:
      raise ValueError(f"Invalid recall method: {method}")

    
