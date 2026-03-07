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

-- Knowledge Graph tables
CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  description TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(user_id, name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(user_id, type);

CREATE TABLE IF NOT EXISTS relationships (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  from_entity_id TEXT NOT NULL,
  to_entity_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  context TEXT,
  memory_id TEXT,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (from_entity_id) REFERENCES entities(id),
  FOREIGN KEY (to_entity_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_relationships_user ON relationships(user_id);
CREATE INDEX IF NOT EXISTS idx_relationships_from ON relationships(from_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_to ON relationships(to_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_memory ON relationships(memory_id);
