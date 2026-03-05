# agent-memory

Persistent memory API for AI agents. Deployed at [memory.coy.gg](https://memory.coy.gg).

## Stack

- **Cloudflare Worker** — Hono-based REST API
- **D1** — SQLite storage for memories
- **Workers AI** — `@cf/baai/bge-small-en-v1.5` embeddings (384-dim)
- **Cosine similarity** — In-worker vector search
- **Claude / Workers AI LLM** — Memory extraction from conversations

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /recall | Semantic memory search |
| POST | /capture | AI-extract + store memories from exchange |
| POST | /store | Directly store a memory |
| GET | /memories | List recent memories |
| DELETE | /memory/:id | Delete a memory |

## Auth

All endpoints (except /health) require `Authorization: Bearer <API_KEY>`.

## Deploy

```bash
npm install
npx wrangler deploy
```

Requires secrets: `API_KEY`, `ANTHROPIC_API_KEY`
