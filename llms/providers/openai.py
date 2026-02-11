from ..base_llm import BaseLLM
from typing import Any
from ..types import InputItem
from openai import OpenAI
import base64


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str,
        provider: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(model, provider, temperature, **kwargs)
        self.client = OpenAI()

    def _prepare_params(
        self,
        llm_input: str | list[InputItem],
        **kwargs: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "input": llm_input,
        }

        if self.temperature:
            params["temperature"] = self.temperature
        if kwargs.get("instructions"):
            params["instructions"] = kwargs.get("instructions")
        if kwargs.get("schema"):
            params["text_format"] = kwargs.get("schema")
        if kwargs.get("tools"):
            params["tools"] = kwargs.get("tools")

        return params

    def call(
        self,
        llm_input: str | list[InputItem],
        **kwargs: Any,
    ) -> str:
        params = self._prepare_params(llm_input, **kwargs)
        response = self.client.responses.parse(**params)
        # For structured output, return the parsed output
        if kwargs.get("schema"):
            return response.output_parsed
        # For image generation, return the image data
        if any(tool.get("type") == "image_generation" for tool in kwargs.get("tools", [])):
            image_data = [
                output.result
                for output in response.output
                if output.type == "image_generation_call"
            ]
            if image_data:
                return {"image_data": base64.b64decode(image_data[0])}
        # For text output, return the text (default)
        return response.output_text
