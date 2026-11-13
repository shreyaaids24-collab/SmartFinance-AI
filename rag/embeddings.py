import hashlib
import math
import re


EMBEDDING_DIMENSION = 384


def embed_texts(texts):
    """Create deterministic local vectors without requiring a model download."""
    vectors = []
    for text in texts:
        vector = [0.0] * EMBEDDING_DIMENSION
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]
        vectors.append(vector)
    return vectors


def get_embedding_model():
    return embed_texts
