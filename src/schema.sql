CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL DEFAULT 'default',
  session_id TEXT,
  scope TEXT NOT NULL DEFAULT 'long_term',
  content TEXT NOT NULL,
  embedding TEXT DEFAULT NULL,
  metadata TEXT DEFAULT '{}',
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(user_id, scope);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
