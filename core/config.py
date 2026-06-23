# core/config.py
# Hybrid Model Configuration - Phase 3
# 70B for smart tasks, 8B for fast tasks

# ── Groq models ───────────────────────────────────────────────
# Quality models (70B) - for tasks needing intelligence
SMART_MODEL = "llama-3.3-70b-versatile"      # Orchestrator, Synthesizer, Refine

# Fast models (8B) - for scoring/grading tasks  
FAST_MODEL = "llama-3.1-8b-instant"          # Reflection, Critic (scoring), Evaluator

# ── Task-specific model assignment ───────────────────────────
ORCHESTRATOR_MODEL = SMART_MODEL   # Smart routing
SYNTHESIZER_MODEL = SMART_MODEL    # Quality answer generation
REFINE_MODEL = SMART_MODEL         # Quality rewrite (Critic refine only)

REFLECTION_MODEL = FAST_MODEL       # Quick relevance scoring
CRITIC_SCORE_MODEL = FAST_MODEL     # Quick draft scoring
EVALUATOR_MODEL = FAST_MODEL        # Quick evaluation

# Backward compatibility (For doc_agent.py and other files)
ACTIVE_MODEL = SMART_MODEL
GENERATOR_MODEL = SMART_MODEL
JUDGE_MODEL = FAST_MODEL

# ── Embeddings ────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# ── Retrieval settings (OPTIMIZED FOR SPEED) ──────────────────
TOP_K_DOCS = 2    # Reduced from 5 - faster retrieval
TOP_K_WEB = 1     # Reduced from 4 - faster web search
CHUNK_SIZE = 512    # Size of each text chunk
CHUNK_OVERLAP = 64  # Overlap between chunks

# ── CRAG threshold ────────────────────────────────────────────
CRAG_THRESHOLD = 0.70

# ── Complexity threshold ──────────────────────────────────────
# Queries scoring BELOW this → simple → skip reflection/critic/eval
# Queries scoring ABOVE this → complex → full pipeline
COMPLEXITY_THRESHOLD = 0.15  # Lowered to ensure business queries trigger full pipeline

# ── Redis memory settings ──────────────────────────────────────
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
MEMORY_TTL = 3600      # 1 hour
MAX_MEMORY_TURNS = 6   # Remember last 6 exchanges