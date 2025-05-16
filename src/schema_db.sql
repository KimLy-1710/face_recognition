-- CREATE EXTENSION vector;

CREATE TABLE face_embeddings (
    id SERIAL PRIMARY KEY,
    person_name VARCHAR(100) NOT NULL,
    person_id VARCHAR(50),
    birthday VARCHAR(50),
    image_path TEXT,
    model VARCHAR(50),
    embedding vector(512)  -- Kích thước vector tùy thuộc vào model của bạn (ArcFace thường là 512)
);

-- Tạo index để tìm kiếm vector nhanh hơn
CREATE INDEX face_embeddings_embedding_idx ON face_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);