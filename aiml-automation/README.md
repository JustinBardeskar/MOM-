# Meeting Intelligence AI/ML Automation

This service owns the workflow beginning at **Meeting Finished / Recording Available**. It provides the stable API boundary used by the company backend and executes the correct route for a recording or direct transcript.

## Implemented processing layer

- Secure streamed download from a short-lived recording or transcript URL.
- Recording path: FFmpeg audio extraction to 16 kHz mono WAV, then Faster-Whisper speech-to-text.
- Direct transcript path: TXT, VTT, or SRT parsing without media processing.
- Shared transcript normalization and a typed unified transcript with timestamps.
- Versioned preprocessing: filler removal, speaker normalization, timestamp cleanup, conservative transcript-noise removal, token-aware chunking, and context bundles.
- Asynchronous local worker and polling API.
- Versioned M2-to-M3 contract that rejects invalid preprocessing before any LLM call.
- Provider-neutral AI Brain with OpenAI, Anthropic Claude, and Google Gemini adapters.
- Model routing, versioned prompts, relevant-context selection, meeting memory, validated-response caching, token/cost tracking, retries, and structured LLM monitoring.
- Meeting understanding followed by ten parallel specialist agents for summaries, actions, decisions, requirements, risks, sentiment, topics, deadlines, questions, and follow-ups.
- Validator, conflict, confidence, and memory-validation stages producing final structured JSON.
- `awaiting_analysis` when Model 3 is disabled and `awaiting_delivery` after Model 3 succeeds.

ERPNext, email, PDF, callback, and storage delivery are not implemented yet. Their boundaries remain represented by injectable ports in `app/domain/ports.py`.

## Local setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8100
```

Open `http://127.0.0.1:8100/docs` for the API contract.

Faster-Whisper downloads the configured model on the first recording job and caches it locally. Use `AUTOMATION_WHISPER_MODEL=tiny.en` for a quick development smoke test and a larger model selected through production accuracy and latency evaluation.

## Enable Model 3

Model 3 is disabled by default so M1 and M2 remain runnable before an LLM key is provided. Configure at least one provider through environment variables, then enable it:

```powershell
$env:AUTOMATION_AI_OPENAI_API_KEY="..."
$env:AUTOMATION_AI_BRAIN_ENABLED="true"
uvicorn app.main:app --reload --port 8100
```

Claude and Gemini use `AUTOMATION_AI_ANTHROPIC_API_KEY` and `AUTOMATION_AI_GEMINI_API_KEY`. Agents never receive these values; only provider adapters inside `app/ai_brain` can access them. See `docs/model3-ai-brain.md` for architecture and configuration.

## Current endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/live` | Process liveness |
| `GET` | `/ready` | Service readiness |
| `GET` | `/health` | Service health |
| `POST` | `/api/v1/meetings/ready` | Accept a finished-meeting event |
| `GET` | `/api/v1/jobs/{job_id}` | Read pipeline path and status |
| `GET` | `/api/v1/jobs/{job_id}/transcript` | Read the unified transcript |
| `GET` | `/api/v1/jobs/{job_id}/preprocessed` | Read cleaned segments, chunks, contexts, and statistics |
| `GET` | `/api/v1/jobs/{job_id}/result` | Read the final structured result |

See `docs/integration-contract.md` for the cross-team payload, sequence, security rules, and production adapter checklist.

## Production note

The current in-memory repository and in-process queue make the complete media path runnable locally. Before deployment, replace them with PostgreSQL and a durable broker so jobs survive restarts and can run on dedicated CPU/GPU workers.
