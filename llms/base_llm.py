from abc import ABC, abstractmethod
from typing import Any

from .types import InputItem


class BaseLLM(ABC):
    def __init__(
        self,
        model: str,
        provider: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ):
        if not model:
            raise ValueError("Model name is required")

        self.model = model
        self.provider = provider
        self.temperature = temperature

    @abstractmethod
    def call(
        self,
        llm_input: str | list[InputItem],
    ) -> str:
        """Call the LLM with the given input.

        Args:
            llm_input: Can be a string or a list of InputItem objects.

        Returns:
            Either a string or any other type of data that the LLM returns.
        """
