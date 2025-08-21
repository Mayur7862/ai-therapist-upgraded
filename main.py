# mvp type app to explore AI-assisted mental health support
import os
import re
from typing import List, Tuple

# load .env if it exists (so you can keep the key in a file)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from google import genai
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

# -------- settings you might tweak later --------
MODEL = "gemini-2.0-flash"
MAX_TURNS_TO_REMEMBER = 4  

# a small persona that keeps the bot kind and non-clinical
PERSONA = """
You are “Aarohi”, an empathetic, non-judgmental friend.
- Validate feelings and ask gentle, open questions.
- Offer only general self-care ideas (breathing, grounding, journaling, simple routines).
- Never diagnose. Never mention or suggest medications or dosages. No clinical treatment.
- Encourage reaching out to trusted people or licensed professionals when appropriate.
- Keep it concise (3–6 short sentences) and end with a gentle question.
""".strip()

# a few safe tips the bot can draw from
SAFE_TIPS = [
    "Box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s — repeat for a minute.",
    "5-4-3-2-1 grounding: 5 things you see, 4 feel, 3 hear, 2 smell, 1 taste.",
    "Quick journal: ‘Right now I feel…’, ‘What I need is…’, ‘One tiny next step is…’.",
    "Sleep basics: regular schedule, dim lights late evening, avoid caffeine ~6h before bed.",
]

# very light guardrails
CRISIS_RE = re.compile(r"\b(kill myself|kill me|end my life|suicide|i want to die|self\s*harm)\b", re.I)
MEDICAL_RE = re.compile(
    r"\b(diagnos(e|is|ing)|dosage|mg|prescrib(e|ed|ing)|antidepressant|benzodiazepine|ssri|snri|xanax|prozac|sertraline|fluoxetine|escitalopram|clonazepam)\b",
    re.I,
)

BOUNDARY_NOTE = (
    "I’m here as a supportive friend, not a professional. "
    "If you need urgent or specialized help, a licensed counselor or doctor is best."
)

INDIA_HELPLINES = [
    ("KIRAN (Govt. of India)", "1800-599-0019"),
    ("AASRA 24/7", "+91-9820466726"),
]

# create the model client
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None


def show_crisis_help() -> None:
    console.print(
        Panel.fit(
            "If you’re in immediate danger, please contact local emergency services.\n\n"
            f"India:\n- {INDIA_HELPLINES[0][0]} — {INDIA_HELPLINES[0][1]}\n"
            f"- {INDIA_HELPLINES[1][0]} — {INDIA_HELPLINES[1][1]}\n\n"
            "You’re not alone. I can stay with you while you reach out.",
            title="Crisis support",
            border_style="red",
        )
    )


def looks_like_crisis(text: str) -> bool:
    return bool(CRISIS_RE.search(text or ""))


def build_prompt(user_text: str, history: List[Tuple[str, str]]) -> str:
    """Simple string prompt: persona + short history + a little hint list."""
    last_turns = history[-MAX_TURNS_TO_REMEMBER :]
    convo = []
    for u, a in last_turns:
        convo.append(f"User: {u}")
        convo.append(f"Friend: {a}")
    convo_block = "\n".join(convo) if convo else "(no prior turns)"

    tips_block = "\n- " + "\n- ".join(SAFE_TIPS)

    return (
        f"SYSTEM PERSONA:\n{PERSONA}\n\n"
        f"RECENT CONVERSATION:\n{convo_block}\n\n"
        f"If useful, you can suggest one of these general ideas:\n{tips_block}\n\n"
        f"User: {user_text}\n\n"
        "Reply now as Aarohi."
    )


def ask_model(prompt: str) -> str:
    """One model call. Keep it boring and reliable."""
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    return getattr(resp, "text", "").strip() or "(no response)"


def lightly_validate(text: str) -> str:
    """
    MVP “validation layer”:
    - If the model accidentally used medical language, ask it to rewrite safely.
    - Otherwise, pass the message through.
    """
    if not text:
        return "I’m here with you. What feels heaviest right now?"

    if MEDICAL_RE.search(text):
        rewrite_rules = (
            "Rewrite this message so it:\n"
            "- uses no medical terms, diagnoses, prescriptions, dosages, or medication names,\n"
            "- keeps an empathetic, friendly tone,\n"
            "- offers only general self-care ideas (breathing, grounding, journaling, routines),\n"
            "- stays concise and ends with a gentle open question.\n\n"
            f"Message:\n{text}\n\n"
            "Return only the rewritten message."
        )
        safe = ask_model(rewrite_rules).strip()
        if not safe:
            safe = (
                "I want to keep you safe. I can’t provide medical advice, "
                "but I can sit with you and explore gentle next steps together."
            )
        return f"{safe}\n\n{BOUNDARY_NOTE}"

    return text


def main() -> None:
    if not client:
        console.print("[bold red]GEMINI_API_KEY is not set. Add it to your shell or .env and rerun.[/bold red]")
        return

    console.print(Panel.fit("AI Therapist (MVP) — type 'exit' to quit", title="Ready", border_style="green"))

    history: List[Tuple[str, str]] = []

    while True:
        console.print()
        user = console.input("[bold cyan]you[/bold cyan]: ").strip()
        if user.lower() in {"exit", "quit", ":q"}:
            break

        # crisis always overrides normal flow
        if looks_like_crisis(user):
            show_crisis_help()
            continue

        # generate reply
        prompt = build_prompt(user, history)
        candidate = ask_model(prompt)

        # quick safety cleanup if needed
        reply = lightly_validate(candidate)

        # show reply
        console.print(Markdown(reply))

        # remember the turn
        history.append((user, reply))

    console.print()
    console.print(Panel.fit("Session ended. Take care. 🌿", border_style="blue"))


if __name__ == "__main__":
    main()
