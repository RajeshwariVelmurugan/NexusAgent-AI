# agents/doc_agent.py
# Fix 1: HuggingFace embeddings — no OpenAI needed
# Fix 2: Groq LLM connected — no MockLLM warning
# Fix 3: Old OpenAI index auto-cleared on mismatch

import os
import shutil
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from core.state import RAGState, AgentResult
from core.config import EMBED_MODEL, TOP_K_DOCS, CHUNK_SIZE, CHUNK_OVERLAP, ACTIVE_MODEL

DOCS_DIR = Path("data/sample_docs")
INDEX_DIR = Path("data/faiss_index")

def _setup_settings():
    """
    Fix 1: HuggingFace embeddings — completely free, runs locally.
    Fix 2: Groq LLM — no more MockLLM warning.
    """
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        embed_batch_size=10,           
        cache_folder="./embedding_cache"  
    )
    print(f"[DocAgent] Embeddings: HuggingFace {EMBED_MODEL} ✓")
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        Settings.llm = Groq(
            model=ACTIVE_MODEL,
            api_key=groq_key,
        )
        print("[DocAgent] LLM: Groq Llama 3.1 ✓")
    else:
        raise EnvironmentError("[DocAgent] GROQ_API_KEY not set in .env")

    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

def _build_index() -> VectorStoreIndex:
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

    print(f"[DocAgent] Indexing {len(documents)} document(s) — first run takes ~2 min...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    print("[DocAgent] Index saved. Next runs will be instant.")
    return index


def _load_index() -> VectorStoreIndex:
    print("[DocAgent] Loading saved index from disk...")
    ctx = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    return load_index_from_storage(ctx)


_index = None


def _get_index():
    global _index
    if _index is None:
        _setup_settings()
        has_index = INDEX_DIR.exists() and any(INDEX_DIR.iterdir())
        _index = _load_index() if has_index else _build_index()
    return _index


def doc_agent_node(state: RAGState) -> dict:
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
        print(f"[DocAgent] Error: {e}")
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
        print(f"  [DocAgent] {score:.3f} | {filename} | {node.text[:60]}...")

    print(f"[DocAgent] {len(results)} result(s) found.")
    return {"agent_results": results}