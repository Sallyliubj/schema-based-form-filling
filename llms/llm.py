from typing import cast, Any

from typing_extensions import Self

from .base_llm import BaseLLM
from .constants import OPENAI_MODELS


class LLM(BaseLLM):
    def __new__(cls, model: str, **kwargs: Any) -> BaseLLM:
        if not model or not isinstance(model, str):
            raise ValueError("Model name is required")

        provider = kwargs.pop("provider", None)
        if not provider:
            provider = cls._infer_provider_from_model(model)

        llm_class = cls._get_provider(provider)

        return cast(Self, llm_class(model=model, provider=provider, **kwargs))

    @classmethod
    def _infer_provider_from_model(cls, model: str) -> str:
        if model in OPENAI_MODELS:
            return "openai"
        else:
            raise ValueError(f"Model {model} is not supported")

    @classmethod
    def _get_provider(cls, provider: str) -> type:
        if provider == "openai":
            from .providers.openai import OpenAILLM

            return OpenAILLM
        elif provider == "azure":
            from .providers.azure import AzureLLM

            return AzureLLM
        else:
            raise ValueError(f"Provider {provider} is not supported")
