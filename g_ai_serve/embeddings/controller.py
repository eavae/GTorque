import abc
import numpy as np
from typing import List, Tuple, TYPE_CHECKING, Dict, Union
from tritonclient.http import InferInput, InferResult
from tritonclient.utils import np_to_triton_dtype
from tokenizers import Encoding
from collections import defaultdict
from collections.abc import Sequence

from g_ai_serve.core import Usage
from g_ai_serve.utils.image_utils import get_image, get_image_resampling, normalize_with
from .data_models import EmbeddingRequest
from .tokenizers import get_tokenizer
from .constants import TEXT_EMBEDDING_MODELS, IMAGE_EMBEDDING_MODELS, RERANKER_MODELS

if TYPE_CHECKING:
    from tokenizers import Tokenizer


def cls_pooling(model_output):
    sentence_embedding = model_output[:, 0]
    return sentence_embedding / np.linalg.norm(
        sentence_embedding, axis=1, keepdims=True
    )


def convert_encoding_to_sparse(encodings: List[Encoding], weights: np.ndarray):
    """
    weights: [batch_size, seq_length, 1]
    """
    results = []
    for i, encoding in enumerate(encodings):
        sparse_matrix = defaultdict(lambda: 0)
        vector = weights[i]
        for is_special, token_id, weight in zip(
            encoding.special_tokens_mask, encoding.ids, np.squeeze(vector)
        ):
            if is_special:
                continue
            sparse_matrix[token_id] = max(sparse_matrix[token_id], weight)
        results.append(sparse_matrix)
    return results


class BaseEmbeddingController:
    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model

    @abc.abstractmethod
    def get_inputs(
        self, request: EmbeddingRequest
    ) -> Tuple[List, List[InferInput], Usage]: ...

    @abc.abstractmethod
    def parse_response(
        self, response: InferResult, encodings: List[Encoding]
    ) -> Union[List[List[float]], List[float]]: ...


class TextEmbeddingController(BaseEmbeddingController):
    SUPPORTED_MODELS = TEXT_EMBEDDING_MODELS

    def get_inputs(
        self, request: EmbeddingRequest
    ) -> Tuple[List, List[InferInput], Usage]:
        tokenizer: Tokenizer = get_tokenizer(self.model)

        encodings = []
        if isinstance(request.input, str):
            encodings.append(tokenizer.encode(request.input))
        elif isinstance(request.input, list):
            encodings = tokenizer.encode_batch(request.input)

        # encodings to triton inputs
        inputs = []

        input_pairs = zip(
            ["input_ids", "token_type_ids", "attention_mask"],
            ["ids", "type_ids", "attention_mask"],
        )
        if request.model == "BAAI__bge-m3":
            input_pairs = zip(
                ["input_ids", "attention_mask"],
                ["ids", "attention_mask"],
            )
        for input_key, attr_key in input_pairs:
            value = np.asarray(
                [getattr(e, attr_key) for e in encodings], dtype=np.int64
            )
            infer_input = InferInput(
                input_key, value.shape, np_to_triton_dtype(value.dtype)
            )
            infer_input.set_data_from_numpy(value)
            inputs.append(infer_input)

        # calculate usage
        prompt_tokens = len([x for e in encodings for x in e.word_ids if x is not None])
        total_tokens = len([x for e in encodings for x in e.attention_mask if x == 1])
        usage = Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens)

        return (encodings, inputs, usage)

    def parse_response(self, response: InferResult, encodings: List[Encoding]) -> Dict:
        if self.model == "BAAI__bge-m3":
            return {
                "dense": response.as_numpy("dense"),
                "sparse": convert_encoding_to_sparse(
                    encodings, response.as_numpy("sparse")
                ),
            }

        hidden_state = response.as_numpy("last_hidden_state")
        return {
            "dense": cls_pooling(hidden_state),
        }


class RerankerController(BaseEmbeddingController):
    SUPPORTED_MODELS = RERANKER_MODELS

    def get_inputs(
        self, request: EmbeddingRequest
    ) -> Tuple[List | List[InferInput] | Usage]:
        tokenizer: Tokenizer = get_tokenizer(self.model)

        # if input is str
        if isinstance(request.input, str):
            raise ValueError("Reranker model does not support single input of string")
        elif isinstance(request.input, Sequence) and isinstance(request.input[0], str):
            encodings = [tokenizer.encode(*request.input)]
        else:
            encodings = tokenizer.encode_batch([tuple(x) for x in request.input])

        # encodings to triton inputs
        inputs = []
        for input_key, attr_key in zip(
            ["input_ids", "attention_mask"],
            ["ids", "attention_mask"],
        ):
            value = np.asarray(
                [getattr(e, attr_key) for e in encodings], dtype=np.int64
            )
            infer_input = InferInput(
                input_key, value.shape, np_to_triton_dtype(value.dtype)
            )
            infer_input.set_data_from_numpy(value)
            inputs.append(infer_input)

        # calculate usage
        prompt_tokens = len([x for e in encodings for x in e.word_ids if x is not None])
        total_tokens = len([x for e in encodings for x in e.attention_mask if x == 1])
        usage = Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens)

        return (encodings, inputs, usage)

    def parse_response(
        self, response: InferResult, encodings: List[Encoding], normalize=True
    ) -> List[float]:
        logits = np.squeeze(response.as_numpy("logits"), axis=-1)
        if normalize:
            return 1 / (1 + np.exp(-logits))
        return logits


class ImageEmbeddingController(BaseEmbeddingController):
    SUPPORTED_MODELS = set(IMAGE_EMBEDDING_MODELS)

    def get_inputs(
        self, request: EmbeddingRequest
    ) -> Tuple[List, List[InferInput], Usage]:
        # 用户的输入是base64或url，需要转换成numpy array
        config = IMAGE_EMBEDDING_MODELS[self.model]
        images = []
        if isinstance(request.input, str):
            images.append(get_image(request.input))
        elif isinstance(request.input, list):
            images = [get_image(url) for url in request.input]

        # 图片转换成numpy array
        images = [
            image.resize(
                config["input_shape"][-2:],
                get_image_resampling(config["interpolation"]),
            ).convert("RGB")
            for image in images
        ]
        images = [np.asarray(image, dtype=np.float32) for image in images]
        images = [
            normalize_with(image, std=config["std"], mean=config["mean"])
            for image in images
        ]
        images = [image.transpose(2, 0, 1) for image in images]
        images = np.stack(images)

        # 构造triton inputs
        infer_input = InferInput(
            "input", images.shape, np_to_triton_dtype(images.dtype)
        )
        infer_input.set_data_from_numpy(images)

        # 计算usage
        prompt_tokens = len(images) * config["tokens"]
        total_tokens = prompt_tokens
        usage = Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens)

        return (images, [infer_input], usage)

    def parse_response(self, response: InferResult) -> np.ndarray:
        config = IMAGE_EMBEDDING_MODELS[self.model]
        output = response.as_numpy(config["output"])

        if config["normalize"]:
            return output / np.linalg.norm(output, axis=1, keepdims=True)

        return output
