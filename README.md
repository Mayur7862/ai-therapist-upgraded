# 🌿 AI Therapist (MVP)

A simple **empathetic friend chatbot** (not clinical) that runs in your terminal.  
Built as a quick prototype using **Python + Gemini API**.

---

## 🚀 Features
- Friendly persona: warm, non-judgmental, supportive.
- Crisis detection: shows helpline numbers immediately.
- Guardrails: blocks medication/diagnosis talk, rewrites unsafe replies.
- Short memory: remembers the last few turns (keeps it fast & cheap).
- Runs fully in the terminal (single file: `main.py`).

---

## 🛠️ Setup
- **Install deps**
  ```bash
  pip install --upgrade pip
  pip install google-genai rich python-dotenv
  ```
- **Set API key**
  - Windows (PowerShell):
    ```powershell
    $env:GEMINI_API_KEY="your-key"
    ```
  - Or create a `.env` file:
    ```
    GEMINI_API_KEY=your-key
    ```

---

## ▶️ Run
```bash
python main.py
```

Type your messages.  
- `exit` or `quit` to end.

---

## 💬 Try these
- “I’ve been anxious all day and can’t sleep.” → supportive tips  
- “Should I take Xanax?” → safe refusal (no med advice)  
- “I want to die.” → crisis helpline panel  

---

## ⚠️ Notes
- This is **not medical advice**.  
- Always reach out to trusted people or licensed professionals if you’re struggling.  
- Crisis helplines are included for India and the US.  
