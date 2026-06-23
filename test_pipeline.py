# test_pipeline.py — run without starting the server
# Usage: python test_pipeline.py

from dotenv import load_dotenv
load_dotenv()

from core.graph import graph
from core.state import RAGState

TESTS = [
    "What are the key highlights from the uploaded documents?",
    "What is the latest news about artificial intelligence in 2026?",
    "Show me the top 5 customers by revenue.",
    "What were our Q3 sales and what do industry reports say about the trend?",
]


def run(query: str):
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)

    result = graph.invoke({
        "query": query,
        "route": [],
        "agent_results": [],
        "final_answer": None,
        "messages": [],
        "metadata": {},
    })

    meta = result.get("metadata", {})
    print(f"\nROUTE     : {result.get('route')}")
    print(f"AGENTS    : {meta.get('agents_used')}")
    print(f"AVG CONF  : {meta.get('avg_confidence')}")
    print(f"SOURCES:")
    for r in sorted(result.get("agent_results", []), key=lambda x: x["confidence"], reverse=True):
        print(f"  [{r['confidence']:.2f}] [{r['agent']}] {r['source']}")
    print(f"\nANSWER:\n{result.get('final_answer')}")


if __name__ == "__main__":
    print("Phase 1 — Groq + Llama 3.1 pipeline test")
    print("Make sure .env has GROQ_API_KEY and PDFs are in data/sample_docs/\n")
    for q in TESTS:
        run(q)
    print("\n" + "="*60)
    print("Done.")