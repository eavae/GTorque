import os
from tritonclient.http.aio import InferenceServerClient
from pydantic import BaseModel


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


triton_client = InferenceServerClient(
    url=os.environ.get("TRITON_SERVER_HTTP_URL", "localhost:8000")
)
