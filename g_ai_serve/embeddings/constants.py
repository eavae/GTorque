TEXT_EMBEDDING_MODELS = {
    "BAAI__bge-large-zh-v1.5",
    "BAAI__bge-m3",
}
TEXT_EMBEDDING_FOLDER = "models/embeddings"
RERANKER_MODELS = {
    "BAAI__bge-reranker-v2-m3",
}

MAX_LENGTH_CONFIG = {
    "BAAI__bge-large-zh-v1.5": 512,
    "BAAI__bge-m3": 8192,
    "BAAI__bge-reranker-v2-m3": 8192,
}

IMAGE_EMBEDDING_MODELS = {
    "laion__CLIP-ViT-bigG-14-laion2B-39B-b160k": {
        "tokens": 256,
        "input": "input",
        "output": "output",
        "input_shape": [3, 224, 224],
        "output_shape": [1024],
        "interpolation": "bicubic",
        "normalize": True,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    }
}
