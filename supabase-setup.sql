-- ============================================================
-- HR-бот на n8n · Налаштування Supabase (pgvector)
-- Урок 4 · Profigent
-- ============================================================
-- Виконати ОДИН раз у вашому Supabase:
--   Dashboard → SQL Editor → New query → вставити весь файл → Run
-- ============================================================

-- 1. Вмикаємо розширення pgvector (дає тип vector і пошук по схожості)
create extension if not exists vector;

-- 2. Таблиця для шматків документів + їх векторів
--    Розмірність 1536 = рівно стільки дає OpenAI text-embedding-3-small.
--    Якщо зміните embedding-модель - зміниться і розмірність (див. примітку внизу).
create table if not exists documents (
  id        bigserial primary key,
  content   text,                    -- текст шматка документа
  metadata  jsonb,                   -- {"source": "Політика відпусток", ...}
  embedding vector(1536)             -- "координати смислу" шматка
);

-- 3. Функція пошуку по схожості (cosine distance).
--    n8n Supabase Vector Store за замовчуванням викликає саме match_documents.
--    Якщо у нодах n8n вказана інша queryName - назва функції має збігатись.
create or replace function match_documents (
  query_embedding vector(1536),
  match_count int default 4,
  filter jsonb default '{}'
) returns table (
  id        bigint,
  content   text,
  metadata  jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where documents.metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- 4. (Рекомендовано) Індекс для швидкого пошуку при великій базі.
--    Для малої бази (до ~1000 шматків) не обов'язково, але не завадить.
--    ivfflat вимагає, щоб у таблиці вже були дані - тому створюйте ПІСЛЯ
--    першого наповнення бази (Workflow 1), або просто пропустіть на старті.
-- create index if not exists documents_embedding_idx
--   on documents using ivfflat (embedding vector_cosine_ops)
--   with (lists = 100);

-- ============================================================
-- ПЕРЕВІРКА після виконання:
--   select * from documents limit 5;   -- має існувати таблиця (поки порожня)
--   select extname from pg_extension where extname = 'vector';  -- має бути 'vector'
-- ============================================================

-- ПРИМІТКА про розмірність:
--   text-embedding-3-small  → 1536  (наш default, найдешевший)
--   text-embedding-3-large  → 3072  (точніше, дорожче)
--   локальна модель (Ollama nomic-embed-text) → 768
-- Розмірність у vector(N) ТУТ і у нодах embeddings n8n МУСИТЬ збігатися.
-- Якщо змінили модель - перестворіть таблицю з правильним N:
--   drop table documents;  -- УВАГА: видаляє всі дані
--   потім заново create table ... vector(N) ...
