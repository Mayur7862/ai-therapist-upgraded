# 🌿 AI Therapist (Enhanced CLI Version)

A **terminal-based AI Therapist app** that acts as a supportive, empathetic friend.  
⚠️ This is **not medical advice** — the system is designed for safe, non-clinical conversations.

---

## 🚀 Features
- **Empathetic Friend Persona**: Non-judgmental, supportive, warm tone.  
- **RAG (Retrieval-Augmented Generation)**: Retrieves safe, curated wellness tips (e.g., stress relief, journaling, breathing).  
- **Multi-hop Reasoning**: Remembers last few turns, connects ideas across inputs.  
- **Safety Guardrails**:  
  - 🚫 Blocks medical/diagnostic advice.  
  - 🛡️ Rewrites unsafe responses automatically.  
  - ☎️ Escalates to helplines on crisis detection.  
- **Evaluation Metrics (prototype)**: Logs safety checks, empathy scores, and context retention.  
- **MVP Simplicity**: Runs in the terminal (single `main.py`).

---

## 🛠️ Setup

### 1. Clone Repo & Enter Folder
```bash
git clone <your-repo-url>
cd ai-therapist
```

### 2. Install Deps
```bash
pip install --upgrade pip
pip install google-genai rich python-dotenv
```

### 3. Set API Key
- **Linux/macOS**
```bash
export GEMINI_API_KEY="your-api-key"
```
- **Windows (PowerShell)**
```powershell
$env:GEMINI_API_KEY="your-api-key"
```
- Or create a `.env` file:
```
GEMINI_API_KEY=your-api-key
```

---

## ▶️ Run
```bash
python main.py
```

- Type your thoughts freely.  
- Use `exit` or `quit` to stop.  

---

## 💬 Example Interactions
- “I’ve been anxious all day and can’t sleep.” → supportive grounding tips  
- “Should I take Xanax?” → polite refusal + suggest professional help  
- “I want to die.” → no AI output, **immediate helpline message**  

---

## 📊 Evaluation (prototype logging)
- **Safety violations caught**  
- **Helpline triggers**  
- **Empathy score approximation** (based on tone heuristics)  
- **Context retention** (tracks if multi-turn context is kept)  

---

## 🛤️ Roadmap
- Add **voice input/output** (multimodal expansion).  
- Safe personalization (store limited preferences).  
- Integration with 3rd-party wellness apps.  
- Federated learning for privacy-preserving improvements.  

---

## ⚠️ Disclaimer
This app is for **wellness support only**.  
It cannot replace professional help. If you’re struggling, always reach out to licensed therapists or call your local helpline.
