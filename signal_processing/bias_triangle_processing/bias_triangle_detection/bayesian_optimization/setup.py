import os
from typing import Optional

from mindfoundry.optaas.client.client import OPTaaSClient


def setup(api_key: Optional[str] = None) -> OPTaaSClient:
    # The URL to the optaas endpoint.
    optimize_url = "https://opt-pre-release.internal.mindfoundry.ai"

    # The API key you got from MindFoundry.
    if api_key is None:
        api_key = os.environ["OPTAAS_API_KEY"]

    # Create the OPTaaS client.
    client = OPTaaSClient(optimize_url, api_key)

    return client
