---
title: "Те саме, але кодом: RAG HR-бот на Python (без n8n)"
lesson: 4
type: handout-bonus
language: ua
purpose: "Показати, що під капотом n8n - звичайний код. Для тих, хто хоче зрозуміти механіку або зробити без візуального конструктора."
---

# Те саме, але кодом: HR-бот на Python

> n8n - це візуальна обгортка над цим самим кодом. Кожна нода = кілька рядків нижче.
> Якщо ви ніколи не писали код - пропустіть цей файл, він не обов'язковий.
> Якщо хочете зрозуміти, ЩО робить n8n всередині - читайте.

## Карта відповідності: нода n8n → код

| Нода n8n | Що робить | Рядок коду |
|---|---|---|
| Default Data Loader | читає файл | `TextLoader(...).load()` |
| Recursive Text Splitter (800/100) | ріже на шматки | `RecursiveCharacterTextSplitter(...)` |
| Embeddings OpenAI | текст → вектор | `OpenAIEmbeddings(...)` |
| Supabase Vector Store (insert) | запис у базу | `SupabaseVectorStore.from_documents(...)` |
| Supabase Vector Store (retrieve) | пошук у базі | `vector_store.similarity_search(...)` |
| AI Agent + GPT-4o-mini | формулює відповідь | `ChatOpenAI(...)` + промпт |
| System Prompt (антигалюцинація) | "тільки з документів" | змінна `SYSTEM_PROMPT` |

---

## 0. Встановлення

```bash
pip install langchain langchain-openai langchain-community supabase
```

Змінні оточення (ті самі ключі, що в credentials n8n):

```bash
export OPENAI_API_KEY="sk-..."
export SUPABASE_URL="https://xxxx.supabase.co"
export SUPABASE_SERVICE_KEY="eyJ..."   # service_role, не anon
```

SQL у Supabase - той самий `supabase-setup.sql`. Код і n8n працюють з ОДНІЄЮ базою.

---

## 1. Наповнення бази знань (= Workflow 1)

```python
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

# Підключення до Supabase (= credential Supabase у n8n)
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

# === Нода "Завантажити HR-документ" ===
docs = TextLoader("Політика-відпусток.md", encoding="utf-8").load()

# === Нода "Розрізати на шматки (800/100)" ===
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Метадані (= поле metadata.source у n8n)
for c in chunks:
    c.metadata["source"] = "Політика відпусток"

# === Нода "Embeddings" + "Записати у Supabase" разом ===
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536 розмірність
SupabaseVectorStore.from_documents(
    chunks,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

print(f"Записано {len(chunks)} шматків у базу.")
# Запускати ОДИН раз на кожен документ. Повторіть для кожного HR-файлу.
```

---

## 2. Сам бот (= Workflow 2)

```python
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

# ВАЖЛИВО: embedding-модель ТА САМА, що при записі. Інакше пошук не працює.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)

# === System Prompt = антигалюцинація (ті самі 5 рядків зі слайда) ===
SYSTEM_PROMPT = """Ти - HR-асистент компанії Profigent.
Відповідай ТІЛЬКИ на основі наданих нижче документів компанії.
Якщо у документах немає відповіді - чесно скажи:
  "Я не знайшов це у документах, зверніться до HR (people@profigent.ai)."
Ніколи не вигадуй правила, цифри чи імена.
Завжди вказуй, з якого документа відповідь (поле source).
Відповідай українською, коротко."""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def ask(question: str) -> str:
    # === Нода retrieve: знайти 4 найближчі шматки ===
    found = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(
        f"[Джерело: {d.metadata.get('source', '?')}]\n{d.page_content}"
        for d in found
    )
    # === Нода AI Agent: відповідь з контексту ===
    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", f"Документи компанії:\n{context}\n\nПитання співробітника: {question}"),
    ]
    return llm.invoke(messages).content

# === Тест (= "момент істини" зі слайда 23) ===
if __name__ == "__main__":
    for q in [
        "Як оформити відпустку?",
        "Скільки днів відпустки мені належить?",
        "Хто підписує заяву на відпустку?",
        "Яка погода завтра в Києві?",   # контрольний: бот має відмовитись
    ]:
        print(f"\nQ: {q}\nA: {ask(q)}")
```

---

## 3. Що цей код доводить

1. **RAG - це не магія, а 2 кроки:** знайти схожі шматки → дати їх моделі з інструкцією "відповідай тільки з них".
2. **n8n робить рівно це**, але без коду - на полотні з нод. Для більшості HR-задач n8n зручніше: видно потік, легко міняти, не треба деплою.
3. **Код потрібен, коли:** дуже нестандартна логіка, інтеграція у власний продукт, або повний контроль (наприклад, локальна модель замість OpenAI - міняєте 2 рядки).

## 4. Локальна модель (приватність, без хмари)

```python
# Замість OpenAI - локальний Ollama (дані не виходять з вашого сервера)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

embeddings = OllamaEmbeddings(model="nomic-embed-text")   # розмірність 768!
llm = ChatOllama(model="llama3.1")
# УВАГА: змініть vector(1536) → vector(768) у supabase-setup.sql
```

> Висновок: і n8n, і код роблять одне. Обирайте n8n для швидкості й наочності,
> код - коли треба вийти за межі конструктора.
