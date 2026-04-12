import os

import openai

from sorrel.buffers import StrBuffer
from sorrel.models import BaseModel

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None

_OPENAI_COMPATIBLE_PROVIDERS: dict = {
    "ollama": {
        "base_url": "http://localhost:11434/v1/",
        "default_api_key": "ollama",
        "env_key": None,
    },
    "openai": {
        "base_url": None,
        "default_api_key": None,
        "env_key": "LLM_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_api_key": None,
        "env_key": "LLM_API_KEY",
    },
}


class Client:
    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        provider: str = "ollama",
        api_key: str | None = None,
    ):
        self.model = model
        self.provider = provider
        self.msg_history = []
        self.max_tokens = max_tokens

        if provider == "anthropic":
            if _anthropic_sdk is None:
                raise ImportError(
                    "anthropic package required for Anthropic provider. "
                    "Install with: pip install anthropic"
                )
            resolved_key = (
                api_key
                or os.environ.get("LLM_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
            )
            self._anthropic_client = _anthropic_sdk.Anthropic(api_key=resolved_key)
            self.client = None
        elif provider in _OPENAI_COMPATIBLE_PROVIDERS:
            cfg = _OPENAI_COMPATIBLE_PROVIDERS[provider]
            env_key_val = os.environ.get(cfg["env_key"]) if cfg["env_key"] else None
            resolved_key = api_key or env_key_val or cfg["default_api_key"]
            kwargs: dict = {"api_key": resolved_key}
            if cfg["base_url"] is not None:
                kwargs["base_url"] = cfg["base_url"]
            self.client = openai.OpenAI(**kwargs)
            self._anthropic_client = None
        else:
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                f"Choose from: {sorted(_OPENAI_COMPATIBLE_PROVIDERS) + ['anthropic']}"
            )

    def clear(self):
        """Clear message history."""
        self.msg_history = []

    def call(
        self,
        prompt: str,
        system_message: str,
        temperature: float = 0.7,
    ):
        self.msg_history += [{"role": "user", "content": prompt}]

        if self.provider == "anthropic":
            response = self._anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_message,
                messages=self.msg_history,
                temperature=temperature,
            )
            content = response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                temperature=temperature,
                max_tokens=self.max_tokens,
                n=1,
            )
            content = response.choices[0].message.content

        self.msg_history += [{"role": "assistant", "content": content}]
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
        provider: str = "ollama",
        api_key: str | None = None,
        **call_params,
    ):
        super().__init__(1, action_space, memory_size)

        self.action_list = action_list
        self.call_params = call_params
        self.model = model_name
        self.instructions = instructions
        self.client = Client(
            model_name, max_tokens=max_tokens, provider=provider, api_key=api_key
        )
        self.stm = ""
        self.memory = StrBuffer(capacity=memory_size, obs_shape=[11, 11])
        self.temperature = temperature

    def take_action(self, state) -> int:

        output = self.client.call(
            prompt=state, system_message=self.instructions, temperature=self.temperature
        )
        response = output.lower()  # type: ignore
        return self.action_list.index(response)

    def format_memories(self, memories):

        states, actions, rewards, dones = memories
        memories = list(zip(states, actions, rewards, dones))
        memories = [
            f"Memory:\n=======\n{state}\nAction: {action}\nReward: {reward}\nDone: {done}\n"
            for state, action, reward, done in memories
        ]
        print(memories[0])

        return memories

    def recall(self, k: int = 1, method: str = "recency"):
        """Recall the `k` most recently observed states and store them in the short-term
        memory buffer.

        Args:
          k (int): The number of memories to recall.
          method (str): The method for recall. By default, recency.
        """
        if method == "recency":
            memories = self.format_memories(
                self.memory[self.memory.idx - k : self.memory.idx]
            )
            self.stm = "\n\n".join(memories)
        elif method == "frequency":
            # TODO: Implement frequency-based recall
            pass
        else:
            raise ValueError(f"Invalid recall method: {method}")
