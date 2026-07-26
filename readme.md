# 🔬 Agent Jailbreak Lab

A local, offline red-teaming environment for probing LLM agent guardrails.
Built with Python, Streamlit, and Ollama.



## Pages

| Page | Description |
|------|-------------|
| 🎯 Attack Runner | Run single attacks or the full deck against a model |
| 📋 Attack Deck | View, edit, add custom attack cases (supports both text and voice decks) |
| 📊 Results Log | Browse all runs with full prompts + responses |
| 📈 Stats | Failure rates by category and model |

## Attack Decks

### Text Attack Deck
Text-based prompt injection and manipulation attacks:
- **Role-play Override** — Persona jailbreaks (DAN, authority impersonation)
- **Prompt Injection** — Malicious payloads in tool output or nested tasks
- **Multi-turn Manipulation** — Gradual context drift across turns
- **Tool Misuse** — Permission escalation via tool calls
- **Encoding Attack** — Instructions hidden in Base64 or other encodings

### Voice Attack Deck
Audio-based attacks targeting voice assistants:
- **Voice Role-play Override** — Spoken persona changes (DAN, authority impersonation)
- **Audio Prompt Injection** — Injections hidden in background speech
- **Third-party Audio** — Tests resistance to nearby speakers or TV
- **Voice Tool Misuse** — Permission escalation via voice commands
- **Authority Attack** — Spoofed administrator commands
- **Voice Authentication** — Speaker verification bypass
- **Conversation Manipulation** — Injections during interruption
- **Encoding Attack** — Voice version of encoded prompt injection
- **Conversation Hijack** — Gradual trust escalation
- **Hidden Audio** — Ultrasonic and imperceptible commands

## Features

- **Dual Deck Selector** — Switch between Text and Voice attack decks in sidebar and Attack Deck page
- **Local or API Inference** — Run attacks against Ollama models, OpenAI-compatible APIs, Anthropic, or Google Gemini
- **Custom Attacks** — Add and test custom attack cases (stored in session)
- **Auto-evaluation** — Keyword-based verdict and scoring system
- **Full Logging** — All runs stored locally with system prompt, attack prompt, and full response

## Architecture

```
app.py              # Streamlit app (single file)
jailbreak_lab.db    # SQLite log (auto-created)
requirements.txt
README.md
```

All data stays local. No API keys required for Ollama runs.

## API-based LLMs

You can also run the lab against remote providers:

- **OpenAI-compatible API**: set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`
- **Anthropic**: set `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, and `ANTHROPIC_MODEL`
- **Google Gemini**: set `GEMINI_API_KEY`, `GEMINI_BASE_URL`, and `GEMINI_MODEL`

Switch to the matching provider in the sidebar to use remote models.