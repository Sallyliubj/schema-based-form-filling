from typing import Literal, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict


class InputItem(TypedDict):
    """Type for formatted LLM messages."""

    role: Literal["user", "assistant"]
    content: str | list[dict[str, str]]


class LLMConfig(BaseModel):
    model: str
    temperature: float
    provider: Optional[str] = None
