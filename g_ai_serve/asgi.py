from fastapi import FastAPI
from typing import Annotated
from fastapi import Depends

from g_ai_serve.security import User, get_current_active_user
from g_ai_serve.embeddings.data_models import (
    EmbeddingRequest,
    Embedding,
    EmbeddingResponse,
    EmbeddingEncodingFormat,
    RerankerRequest,
    RerankerResponse,
    SparseEmbedding,
)
from g_ai_serve.core import triton_client

from g_ai_serve.embeddings.controller import (
    TextEmbeddingController,
    ImageEmbeddingController,
    RerankerController,
)

app = FastAPI()


@app.post("/v1/embeddings")
async def text_embeddings(
    _: Annotated[User, Depends(get_current_active_user)],
    request: EmbeddingRequest,
):
    if request.model in TextEmbeddingController.SUPPORTED_MODELS:
        controller = TextEmbeddingController(request.model)
    elif request.model in ImageEmbeddingController.SUPPORTED_MODELS:
        controller = ImageEmbeddingController(request.model)
    else:
        raise ValueError(f"Unsupported model: {request.model}")

    encodings, inputs, usage = controller.get_inputs(request)

    # call triton
    response = await triton_client.infer(
        model_name=request.model,
        inputs=inputs,
    )
    result_dict = controller.parse_response(response, encodings)

    # format dense response
    dense_embeddings = []
    for i, e in enumerate(result_dict["dense"]):
        if request.encoding_format == EmbeddingEncodingFormat.FLOAT:
            dense_embeddings.append(Embedding(embedding=e.tolist(), index=i))
        elif request.encoding_format == EmbeddingEncodingFormat.BASE64:
            dense_embeddings.append(
                Embedding(embedding=e.tobytes().decode("utf-8"), index=i)
            )

    # format sparse response
    if "sparse" in result_dict:
        sparse_embeddings = []
        for i, e in enumerate(result_dict["sparse"]):
            sorted_tuple = sorted(e.items(), key=lambda x: x[1], reverse=True)
            sorted_tuple = sorted_tuple[: request.sparse_top_n]
            target_indices = set(x[0] for x in sorted_tuple)
            indices = [k for k, v in e.items() if k in target_indices]
            values = [v for k, v in e.items() if k in target_indices]

            sparse_embeddings.append(SparseEmbedding(indices=indices, values=values))

    return EmbeddingResponse(
        model=request.model,
        data=dense_embeddings,
        sparse=sparse_embeddings,
        usage=usage,
    )


@app.post("/v1/rerankings")
async def text_rerankings(
    _: Annotated[User, Depends(get_current_active_user)],
    request: RerankerRequest,
):
    if request.model in RerankerController.SUPPORTED_MODELS:
        controller = RerankerController(request.model)
    else:
        raise ValueError(f"Unsupported model: {request.model}")

    encodings, inputs, usage = controller.get_inputs(request)

    # call triton
    response = await triton_client.infer(
        model_name=request.model,
        inputs=inputs,
    )
    results = controller.parse_response(response, encodings, request.normalized)

    return RerankerResponse(
        model=request.model,
        data=results,
        usage=usage,
    )


@app.get("/heartbeat")
def heartbeat():
    return "OK"
