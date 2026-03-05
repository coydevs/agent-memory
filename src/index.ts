import { Hono } from 'hono';

export interface Env {
  DB: D1Database;
  AI: Ai;
  VECTORIZE: VectorizeIndex;
  API_KEY: string;
  ANTHROPIC_API_KEY: string;
}

interface Memory {
  id: string;
  user_id: string;
  session_id: string | null;
  scope: string;
  content: string;
  metadata: string;
  created_at: number;
  updated_at: number;
}

function generateId(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
}

async function getEmbedding(env: Env, text: string): Promise<number[]> {
  const result = await env.AI.run('@cf/baai/bge-small-en-v1.5', { text: [text] }) as { data: number[][] };
  return result.data[0];
}

async function extractMemoriesWithAnthropic(env: Env, userMessage: string, agentResponse: string): Promise<Array<{content: string, scope: string, metadata: Record<string, unknown>}>> {
  const prompt = `Extract 0-5 factual memories worth storing long-term from this conversation exchange. Only extract concrete facts, preferences, decisions, or context that would be useful in future sessions. Skip pleasantries, trivial remarks, and things already obvious. Return ONLY a valid JSON array (no markdown, no explanation): [{"content": string, "scope": "long_term" or "short_term", "metadata": {}}]

User: ${userMessage}
Agent: ${agentResponse}`;

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-3-5-haiku-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: prompt }],
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Claude API error ${response.status}: ${errText}`);
  }

  const data = await response.json() as { content: Array<{ text: string }> };
  const text = data.content[0]?.text?.trim() || '[]';

  const clean = text.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '').trim();
  return JSON.parse(clean);
}

async function extractMemoriesWithWorkersAI(env: Env, userMessage: string, agentResponse: string): Promise<Array<{content: string, scope: string, metadata: Record<string, unknown>}>> {
  const result = await env.AI.run('@cf/meta/llama-3-8b-instruct', {
    prompt: `<s>[INST] Extract factual memories from this conversation. Return ONLY a JSON array. No text before or after.

Example output: [{"content":"User prefers dark mode","scope":"long_term","metadata":{}}]
If nothing to extract: []

Conversation:
User: ${userMessage}
Agent: ${agentResponse}

JSON array: [/INST]`,
    max_tokens: 512,
    stream: false,
  }) as { response: string };

  const text = '[' + (result.response?.trim() || ']');
  const match = text.match(/\[[\s\S]*?\]/);
  if (!match) return [];
  try {
    return JSON.parse(match[0]);
  } catch {
    return [];
  }
}

function heuristicExtract(userMessage: string, agentResponse: string): Array<{content: string, scope: string, metadata: Record<string, unknown>}> {
  const text = userMessage.trim();
  if (!text || text.length < 10) return [];

  const prefPatterns = [
    /i prefer (.+)/i,
    /i (always|usually|typically) (.+)/i,
    /i (use|like|love|hate|dislike) (.+)/i,
    /my (.+) is (.+)/i,
    /i('m| am) (.+)/i,
    /i (work|live|study) (.+)/i,
  ];

  const memories: Array<{content: string, scope: string, metadata: Record<string, unknown>}> = [];

  for (const pattern of prefPatterns) {
    if (pattern.test(text) && memories.length < 3) {
      const sentence = text.split(/[.!?]/)[0].trim();
      if (sentence.length > 5) {
        memories.push({ content: sentence, scope: 'long_term', metadata: { source: 'heuristic' } });
        break;
      }
    }
  }

  return memories;
}

async function extractMemories(env: Env, userMessage: string, agentResponse: string): Promise<{memories: Array<{content: string, scope: string, metadata: Record<string, unknown>}>, debug?: string}> {
  try {
    const memories = await extractMemoriesWithAnthropic(env, userMessage, agentResponse);
    if (memories.length > 0) return { memories };
  } catch (anthropicErr) {
    console.warn('Anthropic extraction failed:', anthropicErr);
  }

  try {
    const memories = await extractMemoriesWithWorkersAI(env, userMessage, agentResponse);
    if (memories.length > 0) return { memories, debug: 'used_workers_ai' };
  } catch (aiErr) {
    console.warn('Workers AI extraction failed:', aiErr);
  }

  const memories = heuristicExtract(userMessage, agentResponse);
  return { memories, debug: memories.length > 0 ? 'used_heuristic' : 'nothing_extracted' };
}

const app = new Hono<{ Bindings: Env }>();

// Auth middleware
app.use('*', async (c, next) => {
  if (c.req.path === '/health') return next();
  const authHeader = c.req.header('Authorization');
  if (!authHeader || authHeader !== `Bearer ${c.env.API_KEY}`) {
    return c.json({ error: 'Unauthorized' }, 401);
  }
  return next();
});

// GET /health
app.get('/health', (c) => {
  return c.json({ status: 'ok', timestamp: Date.now() });
});

// POST /store — directly store a memory without AI extraction
app.post('/store', async (c) => {
  const { content, user_id = 'default', session_id, scope = 'long_term', metadata = {} } = await c.req.json<{
    content: string;
    user_id?: string;
    session_id?: string;
    scope?: string;
    metadata?: Record<string, unknown>;
  }>();

  if (!content?.trim()) return c.json({ error: 'content is required' }, 400);

  const id = generateId();
  const now = Date.now();

  // Store in D1 (source of truth)
  await c.env.DB.prepare(
    `INSERT INTO memories (id, user_id, session_id, scope, content, metadata, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
  ).bind(id, user_id, session_id || null, scope, content, JSON.stringify(metadata), now, now).run();

  // Embed and upsert to Vectorize
  try {
    const embedding = await getEmbedding(c.env, content);
    await c.env.VECTORIZE.upsert([{
      id,
      values: embedding,
      metadata: { user_id, content, scope, session_id: session_id || '' },
    }]);
  } catch (e) {
    console.error('Vectorize upsert failed:', e);
  }

  return c.json({ stored: true, id, content, scope });
});

// POST /recall
app.post('/recall', async (c) => {
  const { query, user_id = 'default', limit = 5 } = await c.req.json<{
    query: string;
    user_id?: string;
    limit?: number;
  }>();

  if (!query) return c.json({ error: 'query is required' }, 400);

  // Embed the query
  const queryEmbedding = await getEmbedding(c.env, query);

  // Query Vectorize with user_id filter
  const vectorResults = await c.env.VECTORIZE.query(queryEmbedding, {
    topK: limit,
    filter: { user_id },
    returnMetadata: 'none',
  });

  if (!vectorResults.matches?.length) {
    return c.json({ memories: [] });
  }

  // Fetch full records from D1 by IDs
  const ids = vectorResults.matches.map(m => m.id);
  const placeholders = ids.map(() => '?').join(', ');
  const rows = await c.env.DB.prepare(
    `SELECT id, user_id, session_id, scope, content, metadata, created_at, updated_at FROM memories WHERE id IN (${placeholders})`
  ).bind(...ids).all<Memory>();

  // Build a score map from Vectorize results
  const scoreMap = new Map(vectorResults.matches.map(m => [m.id, m.score]));

  const memories = (rows.results || [])
    .map(m => ({
      id: m.id,
      content: m.content,
      scope: m.scope,
      metadata: JSON.parse(m.metadata || '{}'),
      score: scoreMap.get(m.id) ?? 0,
      created_at: m.created_at,
    }))
    .sort((a, b) => b.score - a.score);

  return c.json({ memories });
});

// POST /capture
app.post('/capture', async (c) => {
  const { user_message, agent_response, user_id = 'default', session_id } = await c.req.json<{
    user_message: string;
    agent_response: string;
    user_id?: string;
    session_id?: string;
  }>();

  if (!user_message || !agent_response) {
    return c.json({ error: 'user_message and agent_response are required' }, 400);
  }

  // Extract memories via Claude / Workers AI fallback
  const { memories: extracted, debug: extractDebug } = await extractMemories(c.env, user_message, agent_response);

  if (!extracted.length) {
    return c.json({ captured: 0, memories: [], debug: extractDebug });
  }

  const savedMemories = [];
  const now = Date.now();

  for (const mem of extracted) {
    if (!mem.content?.trim()) continue;

    const id = generateId();

    // Store in D1 (source of truth)
    await c.env.DB.prepare(
      `INSERT INTO memories (id, user_id, session_id, scope, content, metadata, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
    ).bind(
      id,
      user_id,
      session_id || null,
      mem.scope || 'long_term',
      mem.content,
      JSON.stringify(mem.metadata || {}),
      now,
      now,
    ).run();

    // Embed and upsert to Vectorize
    try {
      const embedding = await getEmbedding(c.env, mem.content);
      await c.env.VECTORIZE.upsert([{
        id,
        values: embedding,
        metadata: {
          user_id,
          content: mem.content,
          scope: mem.scope || 'long_term',
          session_id: session_id || '',
        },
      }]);
    } catch (e) {
      console.error('Vectorize upsert failed for memory:', id, e);
    }

    savedMemories.push({ id, content: mem.content, scope: mem.scope || 'long_term', metadata: mem.metadata || {} });
  }

  return c.json({ captured: savedMemories.length, memories: savedMemories });
});

// POST /reindex — re-upsert all D1 memories to Vectorize (admin, one-time migration)
app.post('/reindex', async (c) => {
  const rows = await c.env.DB.prepare(
    'SELECT id, user_id, session_id, scope, content, metadata FROM memories ORDER BY created_at ASC'
  ).all<Memory>();

  let upserted = 0;
  let failed = 0;

  for (const m of rows.results || []) {
    try {
      const embedding = await getEmbedding(c.env, m.content);
      await c.env.VECTORIZE.upsert([{
        id: m.id,
        values: embedding,
        metadata: {
          user_id: m.user_id,
          content: m.content,
          scope: m.scope,
          session_id: m.session_id || '',
        },
      }]);
      upserted++;
    } catch (e) {
      console.error('Reindex failed for', m.id, e);
      failed++;
    }
  }

  return c.json({ reindexed: upserted, failed, total: rows.results?.length ?? 0 });
});

// DELETE /memory/:id
app.delete('/memory/:id', async (c) => {
  const id = c.req.param('id');

  // Delete from D1
  await c.env.DB.prepare('DELETE FROM memories WHERE id = ?').bind(id).run();

  // Delete from Vectorize
  try {
    await c.env.VECTORIZE.deleteByIds([id]);
  } catch (e) {
    console.error('Vectorize delete failed:', e);
  }

  return c.json({ deleted: true });
});

// GET /memories
app.get('/memories', async (c) => {
  const user_id = c.req.query('user_id') || 'default';
  const scope = c.req.query('scope');
  const limit = parseInt(c.req.query('limit') || '20', 10);

  let sql = 'SELECT id, user_id, session_id, scope, content, metadata, created_at, updated_at FROM memories WHERE user_id = ?';
  const params: (string | number)[] = [user_id];

  if (scope) {
    sql += ' AND scope = ?';
    params.push(scope);
  }

  sql += ' ORDER BY created_at DESC LIMIT ?';
  params.push(limit);

  const rows = await c.env.DB.prepare(sql).bind(...params).all<Memory>();

  const memories = (rows.results || []).map(m => ({
    ...m,
    metadata: JSON.parse(m.metadata || '{}'),
  }));

  return c.json({ memories, total: memories.length });
});

export default app;
