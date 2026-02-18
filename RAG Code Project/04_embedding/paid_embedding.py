"""
============================================================
PAID EMBEDDING MODELS – TOP 7 (PRODUCTION READY)
============================================================

• Used by real companies
• Best for RAG applications
• Each model has a separate function
• Correct imports & SDK usage
============================================================
"""

from typing import List
import os


# ============================================================
# 1. OPENAI – TEXT-EMBEDDING-3-LARGE
# ============================================================

def embed_openai_large(texts: List[str]) -> List[List[float]]:
    """
    MODEL:
        text-embedding-3-large
    DIMENSION:
        3072
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )

    return [item.embedding for item in response.data]


# ============================================================
# 2. OPENAI – TEXT-EMBEDDING-3-SMALL
# ============================================================

def embed_openai_small(texts: List[str]) -> List[List[float]]:
    """
    MODEL:
        text-embedding-3-small
    DIMENSION:
        1536
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [item.embedding for item in response.data]


# ============================================================
# 3. COHERE EMBEDDINGS (ENGLISH / MULTILINGUAL)
# ============================================================

def embed_cohere(
    texts: List[str],
    multilingual: bool = False
) -> List[List[float]]:
    """
    MODELS:
        embed-english-v3.0
        embed-multilingual-v3.0
    DIMENSION:
        1024
    """
    import cohere

    client = cohere.Client(os.getenv("COHERE_API_KEY"))

    model_name = (
        "embed-multilingual-v3.0"
        if multilingual
        else "embed-english-v3.0"
    )

    response = client.embed(
        texts=texts,
        model=model_name,
        input_type="search_document"
    )

    return response.embeddings


# ============================================================
# 4. GOOGLE VERTEX AI – TEXT EMBEDDINGS
# ============================================================

def embed_google_vertex(texts: List[str]) -> List[List[float]]:
    """
    MODEL:
        textembedding-gecko
    DIMENSION:
        768
    """
    import vertexai
    from vertexai.preview.language_models import TextEmbeddingModel

    vertexai.init()  # Uses default GCP project & credentials

    model = TextEmbeddingModel.from_pretrained(
        "textembedding-gecko"
    )

    embeddings = model.get_embeddings(texts)
    return [e.values for e in embeddings]


# ============================================================
# 5. AZURE OPENAI EMBEDDINGS
# ============================================================

def embed_azure_openai(texts: List[str]) -> List[List[float]]:
    """
    MODEL:
        text-embedding-3-large (Azure Hosted)
    DIMENSION:
        3072
    """
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )

    return [item.embedding for item in response.data]


# ============================================================
# 6. VOYAGE AI EMBEDDINGS
# ============================================================

def embed_voyage(texts: List[str]) -> List[List[float]]:
    """
    MODEL:
        voyage-large-2
    DIMENSION:
        1536
    """
    import voyageai

    voyageai.api_key = os.getenv("VOYAGE_API_KEY")

    response = voyageai.embed(
        texts=texts,
        model="voyage-large-2"
    )

    return response.embeddings


# ============================================================
# 7. AMAZON BEDROCK – TITAN EMBEDDINGS
# ============================================================

def embed_amazon_titan(text: str) -> List[float]:
    """
    MODEL:
        amazon.titan-embed-text-v1
    DIMENSION:
        1024

    NOTE:
        Bedrock currently embeds one text per request
    """
    import boto3
    import json

    client = boto3.client("bedrock-runtime")

    body = json.dumps({
        "inputText": text
    })

    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=body,
        accept="application/json",
        contentType="application/json"
    )

    result = json.loads(response["body"].read())
    return result["embedding"]


# ============================================================
# END OF FILE
# ============================================================