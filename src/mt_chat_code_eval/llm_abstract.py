# Description: Abstract class for LLM inference
from abc import ABC, abstractmethod
from typing import List


# Base class for LLM inference
class LLM(ABC):
    def __init__(self, model_name: str):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @abstractmethod
    def start_conversation(self, prompt: str) -> str:
        pass

    @abstractmethod
    def continue_conversation(self, prompt: str) -> str:
        pass

    @abstractmethod
    def get_current_conversation(self) -> List[str]:
        pass
