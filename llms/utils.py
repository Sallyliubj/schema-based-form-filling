import os
from collections.abc import Callable

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

AZURE_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


def get_azure_token_provider(client_id: str | None = None) -> Callable[[], str]:
    """
    Get Azure AD token provider for Managed Identity authentication.

    Args:
        client_id: Optional client ID for User-Assigned Managed Identity.
                   If None, uses System-Assigned Managed Identity.

    Returns:
        A callable that returns a bearer token string.
    """
    credential = (
        DefaultAzureCredential(managed_identity_client_id=client_id)
        if client_id
        else DefaultAzureCredential()
    )

    return get_bearer_token_provider(credential, AZURE_COGNITIVE_SERVICES_SCOPE)
