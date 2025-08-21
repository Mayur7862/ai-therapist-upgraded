import os, re, json, time, glob, datetime
from typing import List, Tuple, Dict, Any, Optional

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import numpy as np
from google import genai
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

console = Console()

# -------------------- Config --------------------
MODEL = "gemini-2.0-flash"
EMBED_MODEL = "gemini-embedding-001"
MAX_TURNS = 6                   # keep cost low but give context
USE_RAG_DEFAULT = True
USE_MULTI_HOP_DEFAULT = True
TOP_K = 3                       # RAG top-k
KB_DIR = "kb"
KB_INDEX_PATH = "kb_index.json"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Persona (non-clinical, empathetic friend)
PERSONA = """
You are “Aarohi”, an empathetic, non-judgmental friend.
- Validate feelings and reflect what you hear.
- Offer only general, non-clinical self-care ideas (breathing, grounding, journaling, simple routines).
- Never diagnose. Never name or suggest medications or dosages. No clinical treatment.
- Encourage reaching out to trusted people or licensed professionals when appropriate.
- Keep it concise (3–6 short sentences) and end with a gentle, open question.
""".strip()

# Safety patterns
CRISIS_RE = re.compile(r"\b(kill myself|kill me|end my life|suicide|i want to die|self\s*harm)\b", re.I)
MEDICAL_RE = re.compile(
    r"\b(diagnos(e|is|ing)|dosage|mg|prescrib(e|ed|ing)|antidepressant|benzodiazepine|ssri|snri|xanax|prozac|sertraline|fluoxetine|escitalopram|clonazepam)\b",
    re.I,
)
BOUNDARY = ("I’m here as a supportive friend, not a professional. "
            "If you need urgent or specialized help, a licensed counselor or doctor is best.")

INDIA_HELPLINES = [("KIRAN (Govt. of India)", "1800-599-0019"), ("AASRA 24/7", "+91-9820466726")]
US_HELPLINE = ("988 Suicide & Crisis Lifeline (US)", "Call/Text 988 or chat 988lifeline.org")

# Client
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

# -------------------- RAG Index --------------------
KB_DOCS: List[Dict[str, Any]] = []
KB_VECS: Optional[np.ndarray] = None

def read_kb_docs() -> List[Dict[str, Any]]:
    docs = []
    for path in glob.glob(os.path.join(KB_DIR, "*.md")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            docs.append({"id": path, "title": os.path.basename(path), "text": txt})
        except Exception:
            pass
    return docs

def build_kb_index() -> int:
    """(Re)build embeddings index and save to KB_INDEX_PATH. Returns #docs."""
    global KB_DOCS, KB_VECS
    docs = read_kb_docs()
    vectors = []
    for d in docs:
        if not d["text"].strip():
            continue
        resp = client.models.embed_content(model=EMBED_MODEL, contents=d["text"])
        vec = np.array(resp.embedding.values, dtype="float32")
        vectors.append(vec.tolist())
    KB_DOCS = docs
    KB_VECS = np.array(vectors, dtype="float32") if vectors else None
    with open(KB_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": EMBED_MODEL, "docs": docs, "vectors": vectors}, f, ensure_ascii=False)
    return len(docs)

def load_kb_index() -> int:
    """Load index if present. Return #docs."""
    global KB_DOCS, KB_VECS
    if not os.path.exists(KB_INDEX_PATH):
        KB_DOCS, KB_VECS = [], None
        return 0
    with open(KB_INDEX_PATH, "r", encoding="utf-8") as f:
        idx = json.load(f)
    KB_DOCS = idx.get("docs", [])
    vecs = idx.get("vectors", [])
    KB_VECS = np.array(vecs, dtype="float32") if vecs else None
    return len(KB_DOCS)

def embed_text(text: str) -> np.ndarray:
    resp = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return np.array(resp.embedding.values, dtype="float32")

def rag_top_k(query: str, k: int = TOP_K) -> List[Tuple[str, str]]:
    if KB_VECS is None or not KB_DOCS:
        return []
    q = embed_text(query)
    qn = q / (np.linalg.norm(q) + 1e-8)
    kn = KB_VECS / (np.linalg.norm(KB_VECS, axis=1, keepdims=True) + 1e-8)
    sims = kn @ qn
    idx = sims.argsort()[-k:][::-1]
    out = []
    for i in idx:
        doc = KB_DOCS[int(i)]
        out.append((doc["title"], doc["text"]))
    return out

# -------------------- Utilities --------------------
def extract_json(s: str) -> Dict[str, Any]:
    """Robustly pull the first JSON object from a string; return {} on failure."""
    try:
        # common case: pure JSON
        return json.loads(s)
    except Exception:
        pass
    # try to find the first {...}
    import re
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}

def show_crisis_help() -> None:
    console.print(Panel.fit(
        "If you’re in immediate danger, please contact local emergency services.\n\n"
        f"India:\n- {INDIA_HELPLINES[0][0]} — {INDIA_HELPLINES[0][1]}\n"
        f"- {INDIA_HELPLINES[1][0]} — {INDIA_HELPLINES[1][1]}\n\n"
        f"US: {US_HELPLINE[0]} — {US_HELPLINE[1]}\n\n"
        "You’re not alone. I can stay with you while you reach out.",
        title="Crisis support", border_style="red"
    ))

def looks_like_crisis(text: str) -> bool:
    return bool(CRISIS_RE.search(text or ""))

# -------------------- Multi-hop pipeline --------------------
def ask_model(contents: str, temperature: float = 0.5) -> Tuple[str, Dict[str, int]]:
    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config={"temperature": temperature}
    )
    text = getattr(resp, "text", "").strip() or "(no response)"
    usage = getattr(resp, "usage_metadata", None) or getattr(resp, "usageMetadata", None) or {}
    return text, {
        "prompt_tokens": getattr(usage, "prompt_token_count", 0) if not isinstance(usage, dict) else usage.get("prompt_token_count", 0),
        "candidates_tokens": getattr(usage, "candidates_token_count", 0) if not isinstance(usage, dict) else usage.get("candidates_token_count", 0),
        "total_tokens": getattr(usage, "total_token_count", 0) if not isinstance(usage, dict) else usage.get("total_token_count", 0),
    }

def classify_intent(user_text: str) -> Dict[str, Any]:
    """Hop 1 — classify risk + topics; JSON out."""
    prompt = f"""
Classify the user's message for a supportive, non-clinical chat assistant.
Return pure JSON with keys:
- crisis (bool),
- medical (bool),  # asks about meds/diagnosis/dosage?
- topics (list of short strings),
- emotions (list of short words),
- ask_clarifying (bool)  # True if a short follow-up question would help

Message: {user_text}
JSON:
""".strip()
    txt, _ = ask_model(prompt, temperature=0.2)
    data = extract_json(txt)
    # fallback safety
    return {
        "crisis": bool(looks_like_crisis(user_text)) if "crisis" not in data else bool(data.get("crisis")),
        "medical": bool(MEDICAL_RE.search(user_text)) if "medical" not in data else bool(data.get("medical")),
        "topics": data.get("topics", []),
        "emotions": data.get("emotions", []),
        "ask_clarifying": bool(data.get("ask_clarifying", False)),
    }

def build_prompt(user_text: str, history: List[Tuple[str, str]], rag_hits: List[Tuple[str, str]]) -> str:
    """Assemble persona + short history + optional RAG snippets."""
    last = history[-MAX_TURNS:]
    convo_lines = []
    for u, a in last:
        convo_lines.append(f"User: {u}")
        convo_lines.append(f"Friend: {a}")
    convo_block = "\n".join(convo_lines) if convo_lines else "(no prior turns)"

    rag_block = "\n\n".join([f"### {t}\n{snip}" for t, snip in rag_hits]) if rag_hits else ""
    rag_text = f"\n\nRAG snippets:\n{rag_block}" if rag_block else ""

    return (
        f"SYSTEM PERSONA:\n{PERSONA}\n\n"
        f"RECENT CONVERSATION:\n{convo_block}{rag_text}\n\n"
        f"User: {user_text}\n\n"
        "Write your reply now as Aarohi."
    )

def plan_reply(user_text: str, cls: Dict[str, Any], hits: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Hop 2 — produce a response plan: reflect, 1–3 suggestions (from RAG), open question."""
    hits_str = "\n\n".join([f"### {t}\n{snip}" for t, snip in hits]) if hits else "(no hits)"
    prompt = f"""
Create a small plan for a non-clinical, empathetic reply.
Use this structure in JSON:
{{
  "reflect": "one or two sentences that validate the user's feelings",
  "suggestions": ["1-2 short, actionable ideas grounded in the snippets (if any)"],
  "question": "one gentle, open question to continue the chat"
}}
Snippets (may be empty):
{hits_str}

User message: {user_text}
JSON:
""".strip()
    txt, _ = ask_model(prompt, temperature=0.2)
    plan = extract_json(txt)
    # ensure keys
    return {
        "reflect": plan.get("reflect", "That sounds heavy, and it makes sense to feel this way."),
        "suggestions": plan.get("suggestions", ["Try slow, steady breathing for a minute."]),
        "question": plan.get("question", "What feels like the smallest next step for you?"),
    }

def write_reply(plan: Dict[str, Any]) -> str:
    """Hop 3 — turn the plan into final text (short, warm, non-clinical)."""
    prompt = f"""
Using this plan, write a concise reply (3–6 short sentences), warm and non-judgmental.
Avoid medical terms, diagnosis, or medications. End with the given open question.

Plan:
{json.dumps(plan, ensure_ascii=False)}

Reply:
""".strip()
    txt, _ = ask_model(prompt, temperature=0.5)
    return txt.strip() or "I’m here with you. What feels heaviest right now?"

def rewrite_safely(text: str) -> str:
    rules = (
        "Rewrite this message so it:\n"
        "- uses no medical terms, diagnoses, prescriptions, dosages, or medication names,\n"
        "- keeps an empathetic, friendly tone,\n"
        "- offers only general self-care ideas (breathing, grounding, journaling, routines),\n"
        "- stays concise and ends with a gentle open question.\n\n"
        f"Message:\n{text}\n\n"
        "Return only the rewritten message."
    )
    txt, _ = ask_model(rules, temperature=0.2)
    return txt.strip() or (
        "I want to keep you safe. I can’t provide medical advice, but I can sit with you and explore gentle next steps together."
    )

# -------------------- Metrics --------------------
class Metrics:
    def __init__(self):
        self.turns = 0
        self.rewrites = 0
        self.api_calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0

    def hrr(self) -> float:
        # Harmful Response Rate: rewrites/turns as a proxy
        return (self.rewrites / max(1, self.turns)) * 100.0

    def add_usage(self, usage: Dict[str, int]):
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("candidates_tokens", 0)

# -------------------- CLI helpers --------------------
def print_help():
    table = Table(title="Commands")
    table.add_column("Command"); table.add_column("What it does")
    table.add_row(":help", "Show this help")
    table.add_row(":rag on|off", "Enable/disable RAG snippets")
    table.add_row(":multi on|off", "Enable/disable multi-hop pipeline")
    table.add_row(":reindex", "Build embeddings index from ./kb/*.md")
    table.add_row(":report", "Show session metrics")
    table.add_row(":quit / :exit", "End the session")
    console.print(table)

# -------------------- Main Loop --------------------
def main():
    if not client:
        console.print("[bold red]GEMINI_API_KEY is not set. Add it to your shell or .env and rerun.[/bold red]")
        return

    use_rag = USE_RAG_DEFAULT
    use_multi = USE_MULTI_HOP_DEFAULT
    kb_count = load_kb_index() if use_rag else 0
    console.print(Panel.fit(
        "AI Therapist — CLI (RAG + Multi-hop + Safety). Type ':help' for commands. Type 'exit' to quit.",
        title="Ready", border_style="green")
    )

    history: List[Tuple[str, str]] = []
    metrics = Metrics()
    session = {
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "kb_docs_loaded": kb_count,
        "use_rag": use_rag,
        "use_multi": use_multi,
        "turns": []
    }

    while True:
        console.print()
        user = console.input("[bold cyan]you[/bold cyan]: ").strip()
        if not user:
            continue

        # Commands
        if user.startswith(":"):
            cmd = user[1:].strip().lower()
            if cmd == "help":
                print_help(); continue
            if cmd.startswith("rag"):
                arg = cmd.split()[-1] if len(cmd.split()) > 1 else ""
                if arg in {"on","off"}:
                    use_rag = (arg == "on")
                    session["use_rag"] = use_rag
                    console.print(f"[green]RAG is now {'ON' if use_rag else 'OFF'}.[/green]")
                else:
                    console.print("[yellow]Use ':rag on' or ':rag off'[/yellow]")
                continue
            if cmd.startswith("multi"):
                arg = cmd.split()[-1] if len(cmd.split()) > 1 else ""
                if arg in {"on","off"}:
                    use_multi = (arg == "on")
                    session["use_multi"] = use_multi
                    console.print(f"[green]Multi-hop is now {'ON' if use_multi else 'OFF'}.[/green]")
                else:
                    console.print("[yellow]Use ':multi on' or ':multi off'[/yellow]")
                continue
            if cmd == "reindex":
                n = build_kb_index()
                session["kb_docs_loaded"] = n
                console.print(f"[green]Rebuilt index for {n} KB docs.[/green]")
                continue
            if cmd == "report":
                rep = {
                    "turns": metrics.turns,
                    "rewrites": metrics.rewrites,
                    "hrr_percent": round(metrics.hrr(), 2),
                    "api_calls": metrics.api_calls,
                    "prompt_tokens": metrics.prompt_tokens,
                    "output_tokens": metrics.output_tokens,
                }
                console.print(Panel.fit(json.dumps(rep, indent=2), title="Session metrics", border_style="blue"))
                continue
            if cmd in {"exit","quit"}:
                break
            console.print("[yellow]Unknown command. Try ':help'[/yellow]")
            continue

        if user.lower() in {"exit","quit"}:
            break

        # Crisis first
        if looks_like_crisis(user):
            show_crisis_help()
            session["turns"].append({"user": user, "crisis": True, "reply": "(crisis panel shown)"})
            continue

        # Turn accounting
        metrics.turns += 1
        turn = {"user": user, "crisis": False, "used_rag": use_rag, "used_multi": use_multi,
                "api_calls": 0, "rewrite": False}

        # RAG
        hits = rag_top_k(user, TOP_K) if use_rag else []

        # Multi-hop pipeline
        if use_multi:
            cls = classify_intent(user); metrics.api_calls += 1; turn["api_calls"] += 1
            # Optional: if classifier thinks a clarifying Q is needed, we still keep response short and ask one gentle Q.
            plan = plan_reply(user, cls, hits); metrics.api_calls += 1; turn["api_calls"] += 1
            reply = write_reply(plan); metrics.api_calls += 1; turn["api_calls"] += 1
        else:
            prompt = build_prompt(user, history, hits)
            reply, usage = ask_model(prompt); metrics.api_calls += 1; turn["api_calls"] += 1
            metrics.add_usage(usage)

        # Safety validation (final gate)
        if MEDICAL_RE.search(reply):
            safe = rewrite_safely(reply); metrics.api_calls += 1; turn["api_calls"] += 1
            reply = f"{safe}\n\n{BOUNDARY}"
            metrics.rewrites += 1

        # Present
        console.print(Markdown(reply))
        history.append((user, reply))
        session["turns"].append({**turn, "reply": reply})

    # Save session log
    session["ended_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    out_path = os.path.join(LOG_DIR, f"session-{int(time.time())}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)
        console.print(Panel.fit(f"Session saved → {out_path}", border_style="blue"))
    except Exception as e:
        console.print(f"[yellow]Could not write log: {e}[/yellow]")

if __name__ == "__main__":
    main()
