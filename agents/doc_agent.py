# agents/doc_agent.py
# Uses HuggingFace Inference API (FREE - no local model!)
# No torch, no transformers downloads - build in 30 seconds!

import os
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter

# KEY CHANGE: Use API instead of local HuggingFace model
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from core.state import RAGState, AgentResult
from core.config import TOP_K_DOCS, CHUNK_SIZE, CHUNK_OVERLAP

DOCS_DIR = Path("data/sample_docs")
INDEX_DIR = Path("data/faiss_index")


def _setup_settings():
    """
    Uses HuggingFace Inference API - FREE, no local model needed!
    No torch, no transformers - Railway builds in 30 seconds!
    """
    
    # Get HuggingFace token from environment
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("[DocAgent] ⚠️ HF_TOKEN not set! Add to Railway variables.")
        print("[DocAgent] Get free token: https://huggingface.co/settings/tokens")
    
    # KEY CHANGE: API-based embeddings (FREE, no local download)
    Settings.embed_model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_token,
        task="feature-extraction"
    )
    print(f"[DocAgent] Embeddings: HuggingFace API (all-MiniLM-L6-v2) ✅")
    print(f"[DocAgent] Token: {'✓ Set' if hf_token else '❌ Missing'}")

    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    Settings.llm = None  # No LLM needed for indexing


def _build_index() -> VectorStoreIndex:
    """Build FAISS index from scratch (first run only)"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("[DocAgent] Reading documents from data/sample_docs/ ...")
    documents = SimpleDirectoryReader(
        str(DOCS_DIR),
        required_exts=[".pdf", ".txt", ".md"],
    ).load_data()

    if not documents:
        raise FileNotFoundError(
            "\n[DocAgent] No documents found in data/sample_docs/\n"
            "→ Add at least one PDF or TXT file and run again."
        )

    print(f"[DocAgent] Indexing {len(documents)} document(s)...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    print("[DocAgent] Index saved. Next runs will be instant.")
    return index


def _load_index() -> VectorStoreIndex:
    """Load existing index from disk (fast path)"""
    print("[DocAgent] Loading saved index from disk...")
    ctx = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    return load_index_from_storage(ctx)


_index = None


def _get_index():
    """Singleton pattern - load once, reuse"""
    global _index
    if _index is None:
        _setup_settings()
        has_index = INDEX_DIR.exists() and any(INDEX_DIR.iterdir())
        _index = _load_index() if has_index else _build_index()
    return _index


def doc_agent_node(state: RAGState) -> dict:
    """LangGraph node for document retrieval"""
    query = state["query"]
    print(f"\n[DocAgent] Searching documents for: '{query}'")

    try:
        index = _get_index()
        retriever = index.as_retriever(similarity_top_k=TOP_K_DOCS)
        nodes = retriever.retrieve(query)
    except FileNotFoundError as e:
        print(str(e))
        return {"agent_results": []}
    except Exception as e:
        print(f"[DocAgent] Error during retrieval: {e}")
        return {"agent_results": []}

    results = []
    for node in nodes:
        score = float(node.score) if node.score else 0.5
        filename = node.metadata.get("file_name", "unknown")
        
        results.append(AgentResult(
            agent="doc_agent",
            content=node.text.strip(),
            source=filename,
            confidence=round(score, 4),
        ))
        print(f"  [DocAgent] score={score:.3f} | {filename} | {node.text[:50]}...")

    print(f"[DocAgent] Returning {len(results)} result(s).")
    return {"agent_results": results}