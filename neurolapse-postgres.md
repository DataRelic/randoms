### install sudo apt install -y git build-essential libpq-dev postgresql-server-dev-14
### add support for vectors
```bash
OSX brew install pgvector

Linux
git clone https://github.com/pgvector/pgvector.git
cd pgvector/
make
sudo make install 
```

### Create DB
```bash
CREATE DATABASE neurolapse_db
    WITH 
    OWNER = neurolapse_dev
    ENCODING = 'UTF8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

-- Grant privileges to neurolapse_dev
\c neurolapse_db
GRANT ALL PRIVILEGES ON DATABASE neurolapse_db TO neurolapse_dev;
GRANT ALL PRIVILEGES ON SCHEMA public TO neurolapse_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO neurolapse_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO neurolapse_dev;

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

-- Create accounts table
CREATE TABLE accounts (
    "id" UUID PRIMARY KEY,
    "createdAt" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    "name" TEXT,
    "username" TEXT,
    "email" TEXT NOT NULL,
    "avatarUrl" TEXT,
    "details" JSONB DEFAULT '{}'::jsonb
);

-- Create rooms table
CREATE TABLE rooms (
    "id" UUID PRIMARY KEY,
    "createdAt" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create participants table
CREATE TABLE participants (
    "userId" UUID REFERENCES accounts("id") ON DELETE CASCADE,
    "roomId" UUID REFERENCES rooms("id") ON DELETE CASCADE,
    PRIMARY KEY ("userId", "roomId")
);

-- Create memories table
CREATE TABLE memories (
    "id" UUID PRIMARY KEY,
    "type" TEXT NOT NULL,
    "createdAt" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    "content" JSONB NOT NULL,
    "embedding" vector(1536),
    "userId" UUID REFERENCES accounts("id") ON DELETE SET NULL,
    "agentId" UUID REFERENCES accounts("id") ON DELETE SET NULL,
    "roomId" UUID REFERENCES rooms("id") ON DELETE SET NULL,
    "unique" BOOLEAN DEFAULT true NOT NULL
);


-- Create indexes for memories table
CREATE INDEX idx_memories_embedding ON memories
    USING hnsw ("embedding" vector_cosine_ops);

CREATE INDEX idx_memories_type_room ON memories("type", "roomId");

-- Create indexes for participants table
CREATE INDEX idx_participants_user ON participants("userId");
CREATE INDEX idx_participants_room ON participants("roomId");
```
`hnsw` Index: The `hnsw` index type is used for approximate nearest neighbor searches, suitable for vector embeddings. Ensure that the `pgvector` extension supports this index type in your PostgreSQL version.


