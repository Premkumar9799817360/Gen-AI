from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import MetadataFilter, MetadataFilters

# Assume index is already created and stored
# index = VectorStoreIndex.from_documents(docs)

# =========================================================
# 1. Top-K Similarity Search
# =========================================================
def top_k_retrieval(index):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5
    )
    return retriever


# =========================================================
# 2. Hybrid Search (Vector + Keyword)
# =========================================================
def hybrid_retrieval(index):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
        vector_store_query_mode="hybrid"
    )
    return retriever


# =========================================================
# 3. Metadata-Filtered Search
# =========================================================
def metadata_filtered_retrieval(index):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="file_type", value="pdf")
        ]
    )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
        filters=filters
    )
    return retriever


# =========================================================
# 4. Multi-Vector Retrieval
# (Handled at indexing time, retrieved normally)
# =========================================================
def multi_vector_retrieval(index):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10
    )
    return retriever


# =========================================================
# 5. Parent-Child Retrieval
# =========================================================
def parent_child_retrieval(index):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever
    )
    return query_engine


# =========================================================
# 6. Re-Ranking (Post Processing)
# =========================================================
def reranked_retrieval(index):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.75)
        ]
    )
    return query_engine


# =========================================================
# Example Usage
# =========================================================
def run_query(query_engine):
    response = query_engine.query("Explain vector databases")
    print(response)


# ============================================================
# 2. UNIVERSAL RETRIEVAL FUNCTION
# ============================================================
def universal_retriever(
    index,
    query: str,
    top_k: int = 5,
    mode: str = "similarity",     # similarity | hybrid
    metadata: dict = None,
    rerank: bool = False
):
    filters = None

    if metadata:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key=k, value=v)
                for k, v in metadata.items()
            ]
        )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        vector_store_query_mode=mode,
        filters=filters
    )

    postprocessors = []
    if rerank:
        postprocessors.append(
            SimilarityPostprocessor(similarity_cutoff=0.75)
        )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=postprocessors
    )

    response = query_engine.query(query)
    return response




from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# =====================================================
# 1. VECTOR DB CONNECTION
# =====================================================
def connect_chroma_langchain(persist_dir, collection_name):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    return vectordb

# =====================================================
# 2. UNIVERSAL RETRIEVER
# =====================================================
def universal_langchain_retriever(
    vectordb,
    query,
    top_k=5,
    metadata=None
):
    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": top_k,
            "filter": metadata
        }
    )

    docs = retriever.get_relevant_documents(query)
    return docs