import { Hono } from 'hono';

export interface Env {
  DB: D1Database;
  AI: Ai;
  VECTORIZE: VectorizeIndex;
  API_KEY: string;
  ANTHROPIC_API_KEY: string;
  OPENAI_API_KEY: string;
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
  const res = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'text-embedding-3-small',
      input: text,
    }),
  });
  const data = await res.json() as any;
  if (!res.ok) throw new Error(`OpenAI embedding error ${res.status}: ${JSON.stringify(data)}`);
  return data.data[0].embedding;
}

// ─── Deduplication-aware store helper ───────────────────────────────────────
// Returns {duplicate:true, existing_id} or {stored:true, id}
async function storeMemoryWithDedup(
  env: Env,
  opts: {
    content: string;
    user_id: string;
    session_id?: string | null;
    scope?: string;
    metadata?: Record<string, unknown>;
  }
): Promise<
  | { duplicate: true; existing_id: string; content: string }
  | { stored: true; id: string; content: string; scope: string }
> {
  const { content, user_id, session_id, scope = 'long_term', metadata = {} } = opts;

  // 1. Embed
  const embedding = await getEmbedding(env, content);

  // 2. Check for semantic duplicate (top-1, same user)
  try {
    const dupCheck = await env.VECTORIZE.query(embedding, {
      topK: 1,
      filter: { user_id },
      returnMetadata: 'none',
    });
    if (dupCheck.matches?.length) {
      const top = dupCheck.matches[0];
      if (top.score > 0.92) {
        return { duplicate: true, existing_id: top.id, content };
      }
    }
  } catch (e) {
    console.warn('Dedup check failed, proceeding with insert:', e);
  }

  // 3. Not a duplicate — insert
  const id = generateId();
  const now = Date.now();

  await env.DB.prepare(
    `INSERT INTO memories (id, user_id, session_id, scope, content, metadata, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
  ).bind(id, user_id, session_id || null, scope, content, JSON.stringify(metadata), now, now).run();

  try {
    await env.VECTORIZE.upsert([{
      id,
      values: embedding,
      metadata: { user_id, content, scope, session_id: session_id || '' },
    }]);
  } catch (e) {
    console.error('Vectorize upsert failed:', e);
  }

  return { stored: true, id, content, scope };
}

// ─── AI extraction helpers ───────────────────────────────────────────────────

async function extractMemoriesWithAnthropic(
  env: Env,
  userMessage: string,
  agentResponse: string
): Promise<Array<{content: string, scope: string, metadata: Record<string, unknown>}>> {
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

// New: extract from messages[] array using OpenAI gpt-4o-mini
async function extractMemoriesFromMessages(
  env: Env,
  messages: Array<{role: string; content: string}>
): Promise<string[]> {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: `Extract 0-5 durable facts from this conversation that are worth remembering long-term. Focus on: explicit preferences, project decisions, important facts about tools/systems, lessons learned. Return ONLY a JSON object with a "facts" key containing an array of strings. Empty array if nothing worth storing.`,
        },
        {
          role: 'user',
          content: JSON.stringify(messages),
        },
      ],
      max_tokens: 500,
      response_format: { type: 'json_object' },
    }),
  });

  const data = await res.json() as any;
  if (!res.ok) throw new Error(`OpenAI chat error ${res.status}: ${JSON.stringify(data)}`);

  try {
    const parsed = JSON.parse(data.choices[0].message.content);
    const facts = parsed.facts || parsed.memories || parsed.items || [];
    if (Array.isArray(facts)) {
      return facts
        .map((item: unknown) =>
          typeof item === 'string' ? item : (item as { content?: string }).content || ''
        )
        .filter(Boolean);
    }
  } catch {
    // fall through
  }
  return [];
}

async function extractMemoriesWithWorkersAI(
  env: Env,
  userMessage: string,
  agentResponse: string
): Promise<Array<{content: string, scope: string, metadata: Record<string, unknown>}>> {
  const messages = [
    { role: 'user', content: userMessage },
    { role: 'assistant', content: agentResponse },
  ];
  const strings = await extractMemoriesFromMessages(env, messages);
  return strings.map(s => ({ content: s, scope: 'long_term', metadata: {} }));
}

function heuristicExtract(
  userMessage: string,
  agentResponse: string
): Array<{content: string, scope: string, metadata: Record<string, unknown>}> {
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

async function extractMemories(
  env: Env,
  userMessage: string,
  agentResponse: string
): Promise<{memories: Array<{content: string, scope: string, metadata: Record<string, unknown>}>, debug?: string}> {
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

// ─── App ─────────────────────────────────────────────────────────────────────

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

// POST /store — directly store a memory with deduplication
app.post('/store', async (c) => {
  const { content, user_id = 'default', session_id, scope = 'long_term', metadata = {} } = await c.req.json<{
    content: string;
    user_id?: string;
    session_id?: string;
    scope?: string;
    metadata?: Record<string, unknown>;
  }>();

  if (!content?.trim()) return c.json({ error: 'content is required' }, 400);

  const result = await storeMemoryWithDedup(c.env, { content, user_id, session_id, scope, metadata });

  if ('duplicate' in result) {
    return c.json({ duplicate: true, existing_id: result.existing_id, content: result.content });
  }

  return c.json({ stored: true, id: result.id, content: result.content, scope: result.scope });
});

// POST /recall
app.post('/recall', async (c) => {
  const { query, user_id = 'default', limit = 5 } = await c.req.json<{
    query: string;
    user_id?: string;
    limit?: number;
  }>();

  if (!query) return c.json({ error: 'query is required' }, 400);

  const queryEmbedding = await getEmbedding(c.env, query);

  const vectorResults = await c.env.VECTORIZE.query(queryEmbedding, {
    topK: limit,
    filter: { user_id },
    returnMetadata: 'none',
  });

  if (!vectorResults.matches?.length) {
    return c.json({ memories: [] });
  }

  const ids = vectorResults.matches.map(m => m.id);
  const placeholders = ids.map(() => '?').join(', ');
  const rows = await c.env.DB.prepare(
    `SELECT id, user_id, session_id, scope, content, metadata, created_at, updated_at FROM memories WHERE id IN (${placeholders})`
  ).bind(...ids).all<Memory>();

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

// POST /recall-and-inject — recall + format as system prompt prefix
app.post('/recall-and-inject', async (c) => {
  const { user_id = 'default', query, limit = 10 } = await c.req.json<{
    user_id?: string;
    query: string;
    limit?: number;
  }>();

  if (!query) return c.json({ error: 'query is required' }, 400);

  const queryEmbedding = await getEmbedding(c.env, query);

  const vectorResults = await c.env.VECTORIZE.query(queryEmbedding, {
    topK: limit,
    filter: { user_id },
    returnMetadata: 'none',
  });

  if (!vectorResults.matches?.length) {
    return c.json({
      system_prefix: '',
      memories: [],
    });
  }

  const ids = vectorResults.matches.map(m => m.id);
  const placeholders = ids.map(() => '?').join(', ');
  const rows = await c.env.DB.prepare(
    `SELECT id, user_id, session_id, scope, content, metadata, created_at, updated_at FROM memories WHERE id IN (${placeholders})`
  ).bind(...ids).all<Memory>();

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

  const bulletList = memories.map(m => `- ${m.content}`).join('\n');
  const system_prefix = memories.length
    ? `## Relevant memories from past sessions:\n${bulletList}`
    : '';

  return c.json({ system_prefix, memories });
});

// POST /capture — extract memories from conversation (supports two formats)
//
// Format A (new, preferred): { user_id, messages: [{role, content}...] }
// Format B (legacy): { user_message, agent_response, user_id, session_id }
app.post('/capture', async (c) => {
  const body = await c.req.json<{
    // Format A
    messages?: Array<{role: string; content: string}>;
    // Format B
    user_message?: string;
    agent_response?: string;
    // Shared
    user_id?: string;
    session_id?: string;
  }>();

  const user_id = body.user_id || 'default';
  const session_id = body.session_id;

  // ── Format A: messages[] array ────────────────────────────────────────────
  if (body.messages && Array.isArray(body.messages)) {
    if (!body.messages.length) {
      return c.json({ extracted: 0, stored: 0, duplicates: 0, memories: [] });
    }

    let memoryStrings: string[] = [];
    try {
      memoryStrings = await extractMemoriesFromMessages(c.env, body.messages);
    } catch (e) {
      console.error('extractMemoriesFromMessages failed:', e);
      return c.json({ extracted: 0, stored: 0, duplicates: 0, memories: [], error: String(e) });
    }

    let stored = 0;
    let duplicates = 0;
    const storedMemories: string[] = [];

    for (const content of memoryStrings) {
      if (!content.trim()) continue;
      const result = await storeMemoryWithDedup(c.env, {
        content,
        user_id,
        session_id,
        scope: 'long_term',
        metadata: { source: 'auto_capture' },
      });
      if ('duplicate' in result) {
        duplicates++;
      } else {
        stored++;
        storedMemories.push(content);
      }
    }

    return c.json({
      extracted: memoryStrings.length,
      stored,
      duplicates,
      memories: storedMemories,
    });
  }

  // ── Format B: legacy user_message / agent_response ────────────────────────
  const { user_message, agent_response } = body;
  if (!user_message || !agent_response) {
    return c.json({ error: 'Provide either messages[] or user_message + agent_response' }, 400);
  }

  const { memories: extracted, debug: extractDebug } = await extractMemories(c.env, user_message, agent_response);

  if (!extracted.length) {
    return c.json({ captured: 0, memories: [], debug: extractDebug });
  }

  const savedMemories = [];
  for (const mem of extracted) {
    if (!mem.content?.trim()) continue;

    const result = await storeMemoryWithDedup(c.env, {
      content: mem.content,
      user_id,
      session_id,
      scope: mem.scope || 'long_term',
      metadata: mem.metadata || {},
    });

    if (!('duplicate' in result)) {
      savedMemories.push({ id: result.id, content: result.content, scope: result.scope, metadata: mem.metadata || {} });
    }
  }

  return c.json({ captured: savedMemories.length, memories: savedMemories, debug: extractDebug });
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

  await c.env.DB.prepare('DELETE FROM memories WHERE id = ?').bind(id).run();

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
