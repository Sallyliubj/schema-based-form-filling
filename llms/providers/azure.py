import logging
import os
import time
from typing import Any
import base64

from openai import OpenAI, AzureOpenAI, RateLimitError

from ..base_llm import BaseLLM
from ..types import InputItem
from ..utils import get_azure_token_provider

logger = logging.getLogger(__name__)


class AzureLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        type: str = "text",
        provider: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, provider, temperature, **kwargs)

        self.type = type
        azure_resource_name = os.getenv("AZURE_RESOURCE_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_client_id = os.getenv("AZURE_CLIENT_ID")

        if not azure_resource_name:
            raise ValueError("AZURE_RESOURCE_NAME environment variable is required")
        if not api_version:
            raise ValueError(
                "AZURE_OPENAI_API_VERSION environment variable is required"
            )
        if not azure_client_id:
            raise ValueError("AZURE_CLIENT_ID environment variable is required")

        token_provider = get_azure_token_provider(azure_client_id)

        if type == "image":
            self.client = OpenAI(
                base_url=f"https://{azure_resource_name}/openai/v1/",
                api_key=token_provider,
                default_headers={"api-version": "preview"}
            )
        else:
            self.client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=f"https://{azure_resource_name}.openai.azure.com/",
                azure_ad_token_provider=token_provider,
            )

    def _prepare_params(
        self,
        llm_input: str | list[InputItem],
        **kwargs: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
        }

        if self.type == "text":
            messages = []
            system_prompt = ""
            for item in llm_input:
                if item.get("role") == "system":
                    system_prompt = item.get("content")

            # Instructions override the system prompt
            messages.append({
                "role": "system",
                "content": kwargs.get("instructions", system_prompt),
            })

            for item in llm_input:
                if item.get("role") == "user":
                    messages.append(item)
            params["messages"] = messages

        elif self.type == "image":
            image_input = ""
            prompt_parts = []
            if kwargs.get("instructions"):
                prompt_parts.append(kwargs.get("instructions"))
            
            for item in llm_input:
                if item.get("role") == "user":
                    content = item.get("content")
                    if isinstance(content, str):
                        prompt_parts.append(content)
                    elif isinstance(content, list):
                        for part in content:
                            if part.get("type") == "input_text":
                                prompt_parts.append(part.get("text"))
                            elif part.get("type") == "input_image":
                                image_input = part.get("image_url")

            params["prompt"] = "\n\n".join(prompt_parts)
            params["image"] = image_input

        if self.temperature:
            params["temperature"] = self.temperature

        return params

    def call(
        self,
        llm_input: str | list[InputItem],
        max_retries: int = 5,
        base_delay: float = 1.0,
        **kwargs: Any,
    ) -> str:
        params = self._prepare_params(llm_input, **kwargs)

        for attempt in range(max_retries):
            try:
                # For structured output, return the parsed output
                if kwargs.get("schema"):
                    response = self.client.chat.completions.parse(
                        **params,
                        response_format=kwargs.get("schema"),
                    )
                    if response.choices[0].message.parsed:
                        return response.choices[0].message.parsed
                    return response.choices[0].message.content
                # For image generation, return the image data
                elif any(tool.get("type") == "image_generation" for tool in kwargs.get("tools", [])):
                    response = self.client.images.generate(
                        **params,
                        response_format="b64_json",
                    )
                    b64_data = response.data[0].b64_json
                    image_bytes = base64.b64decode(b64_data)
                    return {"image_data": image_bytes}
                # For text output, return the text (default)
                else:
                    response = self.client.chat.completions.create(**params)
                    return response.choices[0].message.content

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
                    )
                    raise e

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error: {e}")
                    continue
                else:
                    logger.error(
                        f"[red]Max retries ({max_retries}) exceeded. Error: {e}[/]",
                    )
                    raise e
