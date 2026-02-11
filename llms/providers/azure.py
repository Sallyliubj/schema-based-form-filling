import logging
import os
import time
from typing import Any
import base64

from openai import AsyncAzureOpenAI, AzureOpenAI, RateLimitError

from ..base_llm import BaseLLM
from ..types import InputItem
from ..utils import get_azure_token_provider

logger = logging.getLogger(__name__)


class AzureLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        provider: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, provider, temperature, **kwargs)

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_client_id = os.getenv("AZURE_CLIENT_ID")

        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
        if not api_version:
            raise ValueError(
                "AZURE_OPENAI_API_VERSION environment variable is required"
            )
        if not azure_client_id:
            raise ValueError("AZURE_CLIENT_ID environment variable is required")

        token_provider = get_azure_token_provider(azure_client_id)

        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
        )
        self.async_client = AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
        )

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

    def call(
        self,
        llm_input: str | list[InputItem],
        max_retries: int = 5,
        base_delay: float = 1.0,
        **kwargs: Any,
    ) -> str:
        params = self._prepare_params(llm_input)

        for attempt in range(max_retries):
            try:
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

            except RateLimitError as e:
                # Check for Retry-After header in the error response
                retry_after = getattr(e, "retry_after", None)
                delay = float(retry_after) if retry_after else base_delay * 2**attempt

                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"[red]Rate limited (429). Max retries ({max_retries}) exceeded.[/]",
                        extra={"console": True},
                    )
                    raise e

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error processing prompt: {llm_input}. Error: {e}")
                    continue
                else:
                    logger.error(
                        f"[red]Max retries ({max_retries}) exceeded for processing prompt: {llm_input}. Error: {e}[/]",
                        extra={"console": True},
                    )
                    raise e
