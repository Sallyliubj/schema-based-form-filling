from typing import Literal, TypeAlias


OpenAIModels: TypeAlias = Literal[
    "gpt-4",
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
]

OPENAI_MODELS: list[OpenAIModels] = [
    "gpt-4",
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "gpt-5.1",
]
