# agents/db_agent.py
# Uses Groq (Llama 3.3 70B) to generate SQL
# Gracefully skips if database is not configured
# Fix: Added duplicate check to prevent same SQL running twice
# Fix: Added SQL prefix cleaning
# Fix: Better safety filter

import os
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.chains import create_sql_query_chain
from core.state import RAGState, AgentResult
from core.config import ACTIVE_MODEL

FORBIDDEN = {"DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE", "ALTER", "CREATE"}


def _is_safe(sql: str) -> bool:
    """
    Better SQL safety check - extracts only the SELECT part
    """
    sql_upper = sql.upper()
    
    # Find where SELECT starts
    select_idx = sql_upper.find("SELECT")
    if select_idx == -1:
        return False
    
    # Take only from SELECT onwards
    cleaned = sql_upper[select_idx:].strip()
    
    # Block only truly dangerous words
    forbidden = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "ALTER", "INSERT"]
    return not any(word in cleaned for word in forbidden)


def _confidence_from_rows(row_count: int) -> float:
    """Convert row count to confidence score"""
    if row_count == 0:
        return 0.1
    if row_count < 5:
        return 0.6
    if row_count < 20:
        return 0.8
    return 0.9


def _check_db_connection(db_url: str) -> bool:
    """Test DB connection before using it"""
    try:
        import psycopg2
        conn = psycopg2.connect(db_url, connect_timeout=3)
        conn.close()
        return True
    except Exception as e:
        print(f"  [DBAgent] Database not connected — skipping DB agent.")
        print(f"  [DBAgent] Reason: {e}")
        return False


def db_agent_node(state: RAGState) -> dict:
    query = state["query"]
    db_url = os.getenv("DATABASE_URL", "").strip()

    print(f"\n[DBAgent] Generating SQL for: '{query}'")

    # ── Fix: Check if DB agent already ran (prevent duplicates) ──
    existing = [r for r in state.get("agent_results", []) 
                if r["agent"] == "db_agent"]
    if existing:
        print("[DBAgent] Already retrieved — skipping duplicate.")
        return {"agent_results": []}

    # ── Skip if no DB URL configured ───────────────────────────
    if not db_url:
        print("[DBAgent] DATABASE_URL not set — skipping DB agent.")
        return {"agent_results": []}

    # ── Test connection before querying ────────────────────────
    if not _check_db_connection(db_url):
        return {"agent_results": []}

    try:
        db = SQLDatabase.from_uri(db_url)
        llm = ChatGroq(
            model=ACTIVE_MODEL,
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY", ""),
        )

        sql_chain = create_sql_query_chain(llm, db)
        generated_sql = sql_chain.invoke({"question": query}).strip()

        # ── Clean LLM prefixes (Fix: SQLQuery:, Question:, etc.) ──
        if "SQLQuery:" in generated_sql:
            generated_sql = generated_sql.split("SQLQuery:")[-1].strip()
        if "SQL:" in generated_sql:
            generated_sql = generated_sql.split("SQL:")[-1].strip()
        if "Question:" in generated_sql and "SELECT" in generated_sql:
            lines = generated_sql.split('\n')
            for line in lines:
                if 'SELECT' in line.upper():
                    generated_sql = line.strip()
                    break

        print(f"  [DBAgent] Generated SQL:\n    {generated_sql}")

        # ── Safety check ────────────────────────────────────────
        if not _is_safe(generated_sql):
            print("  [DBAgent] Unsafe SQL detected — blocked.")
            return {"agent_results": []}

        # ── Execute query ───────────────────────────────────────
        raw_result = db.run(generated_sql)
        rows_text = str(raw_result).strip()
        row_count = len(rows_text.split("\n")) if rows_text else 0
        confidence = _confidence_from_rows(row_count)

        # Hide credentials from source label
        safe_db = db_url.split("@")[-1] if "@" in db_url else db_url

        print(f"  [DBAgent] {row_count} row(s) returned. confidence={confidence}")

        return {
            "agent_results": [AgentResult(
                agent="db_agent",
                content=f"Results:\n{rows_text}",
                source=f"PostgreSQL — {safe_db}",
                confidence=confidence,
            )]
        }

    except Exception as e:
        print(f"  [DBAgent] Unexpected error: {e}")
        return {"agent_results": []}