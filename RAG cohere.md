# 🔍 Cohere RAG — The Complete Deep Tutorial
### Retrieval-Augmented Generation — From Zero to Production

---

## 📌 Table of Contents

1. [What is RAG? — The Problem it Solves](#1-what-is-rag)
2. [How RAG Works — The Full Pipeline](#2-how-rag-works)
3. [The 3 Core Cohere Tools in RAG](#3-three-tools)
4. [Step 1 — Chunking Your Documents](#4-chunking)
5. [Step 2 — Embedding with Cohere](#5-embedding)
6. [Step 3 — Storing Vectors (Vector Databases)](#6-vector-db)
7. [Step 4 — Searching (Semantic Search)](#7-searching)
8. [Step 5 — Reranking](#8-reranking)
9. [Step 6 — Generating the Answer](#9-generating)
10. [Putting it ALL Together — Full RAG System](#10-full-system)
11. [RAG with Cohere's Built-in Connector](#11-connector)
12. [Common Mistakes and How to Fix Them](#12-mistakes)
13. [Critical Thinking — Advanced Questions](#13-critical)

---

## 1. What is RAG? — The Problem it Solves <a name="1-what-is-rag"></a>

### 📖 Theory

Before you learn RAG, you need to understand the **problem** it solves.

Imagine you are a doctor. You studied medicine for 10 years (training). You know a lot. But now there is a **new drug released in 2025**. You never studied it. What do you do?

You **look it up** in a book, read the relevant pages, and then answer the patient's question using what you just read.

That is exactly what RAG does for an AI model.

**The 3 problems LLMs have without RAG:**

| Problem | What happens | Example |
|---|---|---|
| **Knowledge cutoff** | The model was trained on old data | "Who won the 2025 World Cup?" — the model doesn't know |
| **Hallucination** | The model invents facts confidently | Asks about your company's policy → model makes up an answer |
| **No private data** | The model never saw your documents | Your 500-page PDF manual → model knows nothing about it |

**RAG solves all three by:**
1. Storing YOUR documents in a searchable database
2. When a user asks a question, finding the relevant pieces
3. Giving those pieces to the LLM as context
4. The LLM answers BASED ON YOUR DOCUMENTS — not its training data

> 💡 **New word — "Hallucination"**: When an AI model invents facts that sound real but are completely wrong. For example, inventing a court case that never happened, or giving a fake drug dosage. It is called hallucination because the model is "seeing" things that don't exist. RAG dramatically reduces this because you give the model real text to base its answer on.

**What if you did NOT use RAG?**

If you build an AI assistant for your company without RAG, the AI will:
- Make up answers about your products
- Give outdated information
- Be unable to answer questions about your internal documents
- Lose the trust of your users very quickly

---

## 2. How RAG Works — The Full Pipeline <a name="2-how-rag-works"></a>

### 📖 Theory

RAG has two main phases:

**Phase 1 — Indexing (you do this once, or when data changes):**

```
Your Documents → Split into Chunks → Embed each Chunk → Store vectors in a database
```

**Phase 2 — Querying (happens every time a user asks a question):**

```
User Question → Embed Question → Search database → Get top chunks → Rerank → LLM generates answer
```

Let's draw this more clearly:

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: INDEXING                        │
│                                                             │
│  📄 PDF / Text / Website                                    │
│         │                                                   │
│         ▼                                                   │
│  ✂️  Chunking  →  ["chunk1", "chunk2", "chunk3", ...]       │
│         │                                                   │
│         ▼                                                   │
│  🧠 Cohere Embed  →  [[0.2, 0.8, ...], [0.1, 0.5, ...]]   │
│         │                                                   │
│         ▼                                                   │
│  🗄️  Vector Database  (Chroma / Pinecone / Qdrant)         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    PHASE 2: QUERYING                        │
│                                                             │
│  👤 User: "What is the refund policy?"                      │
│         │                                                   │
│         ▼                                                   │
│  🧠 Cohere Embed  →  [0.3, 0.7, 0.1, ...]                  │
│         │                                                   │
│         ▼                                                   │
│  🔍 Search Vector DB  →  Top 10 similar chunks              │
│         │                                                   │
│         ▼                                                   │
│  🎯 Cohere Rerank  →  Top 3 most relevant chunks            │
│         │                                                   │
│         ▼                                                   │
│  💬 Cohere Command  →  "Our refund policy is 30 days..."    │
└─────────────────────────────────────────────────────────────┘
```

Each step matters. We will now go through every single one, deeply.

---

## 3. The 3 Core Cohere Tools in RAG <a name="3-three-tools"></a>

### 📖 Theory

Cohere gives you exactly the 3 tools you need for RAG:

| Tool | Role in RAG | What it does |
|---|---|---|
| **Embed** | Indexing + Querying | Turns text into vectors |
| **Rerank** | Querying | Re-orders results by relevance |
| **Command** | Querying | Generates the final answer |

> 💡 **Why does Cohere have all 3?** Most providers only give you the LLM (like GPT). Cohere was built from the ground up for enterprise search and RAG, so they built all the tools you need in one place. This is a big advantage — you don't need to mix and match from different companies.

**What if you skip Rerank and just use Embed for search?**

Embedding search (called "dense retrieval") is very good but not perfect. It sometimes retrieves documents that are *related* but not *directly relevant*. Reranking is a second, more powerful filter. Skipping it reduces your answer quality by 20-40% in most real systems.

---

## 4. Step 1 — Chunking Your Documents <a name="4-chunking"></a>

### 📖 Theory

You cannot embed a 100-page PDF as one piece. Here's why:

1. **Embedding models have a token limit** — they can only process a limited amount of text at once
2. **Search quality drops** — if a chunk is too big, the vector becomes a "blurry" average of many topics. When you search for one topic, it might not match well
3. **LLM context is expensive** — you can't give the LLM 100 pages of context every time

So you split your document into small **chunks** — pieces of text, usually 200-500 words each.

> 💡 **New word — "Chunk"**: A chunk is just a small piece of your document. Like cutting a book into small paragraphs and numbering them. Each chunk gets its own vector. When you search, you find the most relevant chunks, not the whole book.

### The 3 main chunking strategies:

**Strategy 1 — Fixed size (simplest)**

Cut every N characters or words, no matter what. Fast but can cut sentences in half.

**Strategy 2 — Sentence-based (better)**

Split at sentence endings (`.`, `!`, `?`). Keeps sentences whole. Better quality.

**Strategy 3 — Overlap chunking (best for most cases)**

Each chunk overlaps slightly with the next. This prevents losing context at the boundary between chunks.

```
Without overlap:
Chunk 1: "The sky is blue. Birds can fly."
Chunk 2: "They migrate south in winter. Temperatures drop."

Problem: If information spans both chunks, you might miss it.

With overlap (50% overlap):
Chunk 1: "The sky is blue. Birds can fly."
Chunk 2: "Birds can fly. They migrate south in winter."
Chunk 3: "They migrate south in winter. Temperatures drop."

Now "Birds can fly and migrate" is captured in chunk 2.
```

### 💻 Code — Chunking

```python
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """
    Split text into overlapping chunks.
    
    chunk_size = how many characters per chunk
    overlap    = how many characters to repeat at the start of each new chunk
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # End of this chunk
        end = start + chunk_size
        
        # Get the chunk
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk.strip())
        
        # Move forward, BUT go back by 'overlap' to create overlap
        start = end - overlap
    
    return chunks


# ---- Test it ----
sample_text = """
Cohere is an AI company founded in 2019. It was created by researchers who
worked at Google Brain. Their main products are language models for enterprise use.
The company focuses on search, summarization, and text generation. Cohere's Embed
model is considered one of the best in the world for semantic search. Their Command
model is optimized for retrieval-augmented generation tasks. Cohere also offers
private deployment, meaning companies can run the models on their own servers.
This is a major advantage for businesses that handle sensitive data.
"""

chunks = chunk_text(sample_text, chunk_size=200, overlap=40)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)
    print(f"Length: {len(chunk)} characters")
```

### 🔍 Code Explanation

```python
while start < len(text):
```
- We keep looping until we have processed the entire text

```python
end = start + chunk_size
chunk = text[start:end]
```
- We cut from position `start` to position `end`
- `text[start:end]` is Python's **slice** syntax — takes a portion of the string

```python
start = end - overlap
```
- Instead of moving forward by the full `chunk_size`, we go back by `overlap`
- This creates the "overlap" — the next chunk starts a bit before where this one ended

**What if you don't chunk and just embed the whole document?**

The Cohere Embed model has a maximum input of ~512 tokens (~380 words). If your document is longer, the text gets cut off silently. You lose information. Always chunk first.

---

## 5. Step 2 — Embedding with Cohere <a name="5-embedding"></a>

### 📖 Theory

An **embedding** is a way to represent the *meaning* of text as a list of numbers (a vector).

Why numbers? Because computers can compare numbers mathematically. Two texts with similar meaning will produce vectors that are *close to each other* in mathematical space.

Think of it like a map:
- "dog" and "puppy" are very close on the map
- "dog" and "quantum physics" are very far apart
- "car" and "automobile" are almost on top of each other

This is called **semantic similarity** — similarity of meaning, not just words.

> 💡 **New word — "Semantic"**: Semantic means *related to meaning*. Semantic search searches by meaning. Semantic similarity means two things have similar meaning. This is different from "lexical" (related to exact words/letters).

### Cohere's Embed model — important details

The `input_type` parameter is critical. You MUST use the right one:

| When you embed | Use `input_type` |
|---|---|
| Documents to store in your database | `"search_document"` |
| A user's search query | `"search_query"` |
| Text you want to classify | `"classification"` |
| Text for clustering | `"clustering"` |

**Why does this matter?** Cohere trains its model to make query vectors and document vectors work well *together*. A query vector is optimized to point toward document vectors. If you use the wrong type, the math breaks and your search quality drops.

### 💻 Code — Embedding Documents

```python
import cohere
import os

co = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY"))

def embed_documents(chunks: list[str]) -> list[list[float]]:
    """
    Take a list of text chunks and return their embedding vectors.
    We batch them to avoid hitting API limits.
    """
    
    # Cohere allows max 96 texts per API call
    # So if we have more, we split into batches
    BATCH_SIZE = 90
    all_vectors = []
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        
        print(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)...")
        
        response = co.embed(
            texts=batch,
            model="embed-v4.0",
            input_type="search_document",   # IMPORTANT: these are documents
            embedding_types=["float"]
        )
        
        # Add this batch's vectors to our list
        all_vectors.extend(response.embeddings.float)
    
    return all_vectors


def embed_query(query: str) -> list[float]:
    """
    Embed a single user query.
    Notice: input_type is "search_query", not "search_document"!
    """
    response = co.embed(
        texts=[query],
        model="embed-v4.0",
        input_type="search_query",   # IMPORTANT: this is a query
        embedding_types=["float"]
    )
    return response.embeddings.float[0]


# ---- Test it ----
chunks = [
    "Cohere was founded in 2019 by Aidan Gomez and other ex-Google researchers.",
    "The Embed v4 model supports both text and images.",
    "Command R is optimized for retrieval-augmented generation.",
]

vectors = embed_documents(chunks)
print(f"\nNumber of vectors: {len(vectors)}")
print(f"Each vector has {len(vectors[0])} numbers")
print(f"First 5 numbers of vector 1: {vectors[0][:5]}")
```

### 🔍 Code Explanation

```python
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i : i + BATCH_SIZE]
```
- `range(0, total, BATCH_SIZE)` creates steps: 0, 90, 180, 270...
- At each step, we take the next 90 chunks
- This is called **batching** — processing in groups to avoid API limits

```python
all_vectors.extend(response.embeddings.float)
```
- `.extend()` adds all items from a list into another list
- (Different from `.append()` which would add the whole list as ONE item)

**What is the vector size?**
Cohere's `embed-v4.0` returns vectors with **1536 numbers** per text. So if you embed 1000 chunks, you get a matrix of 1000 × 1536 numbers. This is what you store in your vector database.

**What if you don't batch and send all at once?**
The API will return an error for large inputs. Always batch. The safe limit is 90 per request.

---

## 6. Step 3 — Storing Vectors (Vector Databases) <a name="6-vector-db"></a>

### 📖 Theory

After embedding your chunks, you have thousands of vectors. You need somewhere to store them AND search them efficiently.

A **vector database** is a special database built for storing and searching vectors quickly.

> 💡 **New word — "Vector Database"**: A normal database (like MySQL) stores rows of data and lets you search by exact match or text keywords. A vector database stores vectors (lists of numbers) and lets you search by *mathematical similarity* — finding the vectors closest to your query vector. It is like finding the nearest neighbor on a map.

### Popular vector databases:

| Database | Best for | Cost | Difficulty |
|---|---|---|---|
| **ChromaDB** | Local development, learning | Free | Very Easy |
| **Qdrant** | Production, self-hosted | Free / Paid cloud | Easy |
| **Pinecone** | Production, fully managed | Free tier + Paid | Easy |
| **Weaviate** | Enterprise, complex setups | Free / Paid | Medium |
| **pgvector** | If you already use PostgreSQL | Free | Medium |

**For learning → use ChromaDB** (runs locally, no signup, no cost)
**For production → use Qdrant or Pinecone**

### 💻 Code — Storing with ChromaDB (Local)

```bash
pip install chromadb
```

```python
import chromadb
import cohere
import os

co = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY"))

# ---- Setup ChromaDB ----
# This creates a local database saved in a folder called "chroma_db"
# All your vectors will be stored here, even after you restart Python
client = chromadb.PersistentClient(path="./chroma_db")

# A "collection" is like a table in a normal database
# It holds all your chunks and their vectors
collection = client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}   # Use cosine similarity for search
)

# ---- Embed and Store ----
def index_documents(chunks: list[str], doc_name: str = "doc"):
    """Embed chunks and store them in ChromaDB."""
    
    print(f"Embedding {len(chunks)} chunks...")
    
    response = co.embed(
        texts=chunks,
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    vectors = response.embeddings.float
    
    # Each document needs a unique ID
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    
    # Store everything in ChromaDB
    collection.add(
        ids=ids,              # Unique identifiers for each chunk
        documents=chunks,     # The original text (so we can retrieve it later)
        embeddings=vectors,   # The vectors (for similarity search)
        metadatas=[{"source": doc_name, "chunk_index": i} for i in range(len(chunks))]
        # metadatas = extra info you can filter by later
    )
    
    print(f"Stored {len(chunks)} chunks successfully!")


# ---- Test it ----
my_chunks = [
    "Cohere was founded in 2019 by Aidan Gomez, Ivan Zhang, and Nick Frosst.",
    "The Command A model is specifically built for agentic AI tasks.",
    "Cohere's Embed v4 model supports both text and images (multimodal).",
    "Cohere offers private deployment for enterprise customers.",
    "Reranking takes a list of results and orders them by relevance to a query.",
    "RAG stands for Retrieval-Augmented Generation.",
    "Vector databases store embeddings and support fast similarity search.",
    "Python was created by Guido van Rossum and first released in 1991.",
]

index_documents(my_chunks, doc_name="cohere_facts")

print(f"\nTotal chunks in database: {collection.count()}")
```

### 🔍 Code Explanation

```python
client = chromadb.PersistentClient(path="./chroma_db")
```
- `PersistentClient` saves your vectors to disk
- So when you close Python and reopen it, the data is still there
- The path `"./chroma_db"` is just a folder that ChromaDB creates automatically

```python
collection.add(
    ids=ids,
    documents=chunks,
    embeddings=vectors,
    metadatas=[...]
)
```
- `ids` — every chunk must have a unique name/id
- `documents` — the original text (stored separately from the vector)
- `embeddings` — the vectors (used for similarity search)
- `metadatas` — extra information (like which file it came from, or its page number)

**What is `hnsw:space: cosine`?**

HNSW is the search algorithm ChromaDB uses. Cosine similarity is the math formula for comparing vectors. It measures the **angle** between two vectors (not the distance). Cohere's embed model works best with cosine similarity.

> 💡 **New word — "HNSW"**: Hierarchical Navigable Small World. It is a very fast algorithm for finding the nearest vectors. Without it, searching 1 million vectors would take forever. HNSW can search 1 million vectors in milliseconds. You don't need to understand how it works internally — just know it makes search fast.

**What if you don't use a vector database and just store in a Python list?**

For 10 chunks — fine. For 10,000 chunks — searching becomes slow (you have to compare every single vector one by one). For 1,000,000 chunks — completely unusable. Vector databases use smart algorithms (like HNSW) to search millions of vectors in milliseconds.

---

## 7. Step 4 — Searching (Semantic Search) <a name="7-searching"></a>

### 📖 Theory

Now that your chunks are stored, you can search. The process:

1. User types a question
2. You embed the question (using `input_type="search_query"`)
3. You send the question vector to the database
4. The database finds the chunks whose vectors are most *similar* to the question vector
5. You get back the top N most relevant chunks

This is called **semantic search** — you are searching by meaning, not by exact keywords.

**Example of why semantic search is powerful:**

- Query: "How much does it cost?"
- Keyword search finds: only documents that contain the word "cost"
- Semantic search finds: documents about "pricing", "fees", "subscription", "plans", "charges" — all related by meaning

### 💻 Code — Searching ChromaDB

```python
def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search for the most relevant chunks for a given query.
    Returns top_k results with their text and similarity scores.
    """
    
    # Step 1: Embed the query
    # IMPORTANT: use "search_query" here, NOT "search_document"
    q_response = co.embed(
        texts=[query],
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    query_vector = q_response.embeddings.float[0]
    
    # Step 2: Search the database
    results = collection.query(
        query_embeddings=[query_vector],   # The question vector
        n_results=top_k,                   # How many results to return
        include=["documents", "metadatas", "distances"]
        # distances = how similar each result is (lower = more similar for cosine)
    )
    
    # Step 3: Format the results nicely
    formatted = []
    for i in range(len(results["documents"][0])):
        formatted.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            # Convert distance to similarity score (0 to 1, higher = better)
            "similarity": 1 - results["distances"][0][i]
        })
    
    return formatted


# ---- Test it ----
query = "What makes Cohere special for enterprise?"
results = semantic_search(query, top_k=5)

print(f"Query: '{query}'")
print(f"\nTop {len(results)} results:\n")
for i, r in enumerate(results):
    print(f"#{i+1} (similarity: {r['similarity']:.3f})")
    print(f"    {r['text']}")
    print()
```

### 🔍 Code Explanation

```python
results = collection.query(
    query_embeddings=[query_vector],
    n_results=top_k,
    include=["documents", "metadatas", "distances"]
)
```
- `query_embeddings` — the vector of the user's question
- `n_results` — how many matching chunks to return
- `include` — what information to include in the response

```python
"similarity": 1 - results["distances"][0][i]
```
- ChromaDB returns `distances` (lower = more similar)
- We convert to `similarity` (higher = more similar) by doing `1 - distance`
- This is easier to think about: `1.0` = perfect match, `0.0` = completely different

**Why get top_k=10 if we only want 3 answers?**

Because the next step — **reranking** — works better with more candidates. Think of it as:
- Embedding search = fast first filter (keeps top 10 out of 1,000,000)
- Reranking = slow but precise second filter (picks best 3 out of 10)

---

## 8. Step 5 — Reranking <a name="8-reranking"></a>

### 📖 Theory

Embedding-based search is fast and finds *generally relevant* results. But it has weaknesses:

- It works at the "semantic neighborhood" level — all 10 results are in the right area
- But which of the 10 is MOST specifically relevant to the exact question?
- Embedding models are optimized for speed, not deep relevance judgment

**Reranking** solves this. It takes your top 10 search results and uses a much more powerful model to read each one carefully and score it against the query.

The reranker reads the query AND the document together, like a human reader would. It is slower but much more accurate.

> 💡 **New word — "Cross-encoder"**: The reranker is a type of model called a cross-encoder. It reads the query and document AT THE SAME TIME, in one pass. This lets it understand the relationship between them deeply. Compare to the embedding model (called a bi-encoder) which embeds query and document separately and just compares the numbers. Cross-encoders are more accurate but slower — that's why we only rerank a small number of candidates.

### 💻 Code — Reranking

```python
def rerank_results(query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
    """
    Take search results and rerank them for better precision.
    
    query      = the user's question
    candidates = results from semantic_search()
    top_n      = how many top results to return after reranking
    """
    
    # Extract just the text from each candidate
    texts = [r["text"] for r in candidates]
    
    # Ask Cohere's reranker to score each one
    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=texts,
        top_n=top_n,
        return_documents=True   # Include the document text in the response
    )
    
    # Build final results with rerank scores
    reranked = []
    for result in response.results:
        reranked.append({
            "text": texts[result.index],              # Original text
            "metadata": candidates[result.index]["metadata"],  # Original metadata
            "original_rank": result.index + 1,        # Position before reranking
            "rerank_score": result.relevance_score,   # New relevance score (0 to 1)
        })
    
    return reranked


# ---- Test it ----
query = "What makes Cohere special for enterprise?"

# First: semantic search (get top 10)
candidates = semantic_search(query, top_k=10)

# Then: rerank (pick best 3)
final_results = rerank_results(query, candidates, top_n=3)

print(f"Query: '{query}'")
print(f"\nTop 3 results after reranking:\n")
for i, r in enumerate(final_results):
    print(f"#{i+1}  Rerank Score: {r['rerank_score']:.4f}  (was rank #{r['original_rank']} before)")
    print(f"     {r['text']}")
    print()
```

### 🔍 Code Explanation

```python
response = co.rerank(
    model="rerank-v3.5",
    query=query,
    documents=texts,
    top_n=top_n,
    return_documents=True
)
```
- `query` — the user's question
- `documents` — the list of candidate texts to score
- `top_n` — return only the top N after scoring
- `return_documents=True` — include the original text in the response

```python
result.relevance_score
```
- A number between 0 and 1
- Higher = more relevant to the query
- The reranker considers things like: does the document ANSWER the question? Or just RELATE to it?

**What if you skip reranking?**

Let's say your embedding search returns these 10 results for "What is the refund policy?":
1. A paragraph about customer satisfaction (related but doesn't answer)
2. The actual refund policy paragraph (the answer)
3. A paragraph about returns (close but not exact)

Without reranking, you might send result #1 to the LLM first. With reranking, result #2 jumps to the top. Your answer quality improves dramatically.

---

## 9. Step 6 — Generating the Answer <a name="9-generating"></a>

### 📖 Theory

Now we have the top 3 most relevant chunks. We give them to the LLM (Cohere Command) as **context** and ask it to answer the user's question based on that context.

This is called **grounded generation** — the answer is grounded in (based on) real text, not invented.

> 💡 **New word — "Grounded"**: In AI, "grounded" means the model's answer is based on specific real text you provided. It is the opposite of hallucination. A grounded answer can always be traced back to a source.

### 💻 Code — Generating the Answer

```python
def generate_answer(query: str, context_chunks: list[dict]) -> dict:
    """
    Given a query and relevant context chunks, generate a grounded answer.
    
    Returns the answer AND the sources it was based on.
    """
    
    # Build the context string
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = chunk["metadata"].get("source", "unknown")
        context_parts.append(f"[Source {i+1}: {source}]\n{chunk['text']}")
    
    context = "\n\n".join(context_parts)
    
    # The prompt — clear instructions for the model
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.

Rules:
- Base your answer ONLY on the context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say: "I don't have enough information to answer this question."
- Be concise and clear.
- After your answer, list which sources you used.

=== CONTEXT ===
{context}

=== QUESTION ===
{query}

=== YOUR ANSWER ==="""
    
    response = co.chat(
        model="command-r-plus",   # Strong model for high-quality answers
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer_text = response.message.content[0].text
    
    return {
        "answer": answer_text,
        "sources": [c["text"] for c in context_chunks],
        "num_sources": len(context_chunks)
    }


# ---- Test it ----
query = "What makes Cohere special for enterprise?"

# Full pipeline
candidates = semantic_search(query, top_k=10)
top_chunks = rerank_results(query, candidates, top_n=3)
result = generate_answer(query, top_chunks)

print(f"Question: {query}")
print(f"\nAnswer:\n{result['answer']}")
print(f"\nBased on {result['num_sources']} sources.")
```

### 🔍 Code Explanation

```python
prompt = f"""...Rules:
- Base your answer ONLY on the context...
- If the context does not contain enough information..."""
```
- We give the model **explicit rules** in the prompt
- The most important rule: answer ONLY from the context
- This is what prevents hallucination — the model is instructed to say "I don't know" instead of making something up

```python
model="command-r-plus"
```
- We use `command-r-plus` here (not the cheap R7B) because:
  - Generating the final answer is the most important step
  - Quality matters here — it's what the user sees
  - Save money on embedding (use cheap batch), spend on the final answer

**What if you don't give clear instructions in the prompt?**
The model might use its training data instead of your context. Or it might mix both. Always be explicit: "answer ONLY from the context below."

---

## 10. Putting it ALL Together — Full RAG System <a name="10-full-system"></a>

### 💻 Complete RAG System — One Clean Class

```python
import cohere
import chromadb
import os

co = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY"))


class CohereRAG:
    """
    A complete RAG system using:
    - Cohere Embed for vectorizing
    - ChromaDB for storage and search
    - Cohere Rerank for precision
    - Cohere Command for answer generation
    """
    
    def __init__(self, collection_name: str = "rag_collection"):
        # Set up local vector database
        self.db = chromadb.PersistentClient(path="./rag_db")
        self.collection = self.db.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"RAG system ready. Documents in DB: {self.collection.count()}")
    
    # ── PHASE 1: INDEXING ──────────────────────────────────────────
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks
    
    def add_document(self, text: str, source_name: str):
        """
        Add a document to the RAG system.
        Automatically chunks and embeds it.
        """
        print(f"\nAdding document: '{source_name}'")
        
        # Step 1: Chunk
        chunks = self.chunk_text(text)
        print(f"  Split into {len(chunks)} chunks")
        
        # Step 2: Embed (in batches of 90)
        all_vectors = []
        BATCH = 90
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i+BATCH]
            resp = co.embed(
                texts=batch,
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"]
            )
            all_vectors.extend(resp.embeddings.float)
        print(f"  Embedded {len(all_vectors)} vectors")
        
        # Step 3: Store
        ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=all_vectors,
            metadatas=[{"source": source_name, "chunk_index": i} for i in range(len(chunks))]
        )
        print(f"  Stored in database. Total documents: {self.collection.count()}")
    
    # ── PHASE 2: QUERYING ──────────────────────────────────────────
    
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Semantic search — find relevant chunks."""
        
        # Embed the query (search_query, not search_document!)
        q_resp = co.embed(
            texts=[query],
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"]
        )
        q_vector = q_resp.embeddings.float[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[q_vector],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i]
            }
            for i in range(len(results["documents"][0]))
        ]
    
    def rerank(self, query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
        """Rerank candidates for better precision."""
        
        if not candidates:
            return []
        
        texts = [c["text"] for c in candidates]
        
        resp = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=texts,
            top_n=min(top_n, len(texts))
        )
        
        return [
            {
                "text": candidates[r.index]["text"],
                "metadata": candidates[r.index]["metadata"],
                "rerank_score": r.relevance_score
            }
            for r in resp.results
        ]
    
    def answer(self, query: str, context_chunks: list[dict]) -> str:
        """Generate a grounded answer from the context."""
        
        if not context_chunks:
            return "I could not find relevant information to answer your question."
        
        context = "\n\n".join([
            f"[Source: {c['metadata'].get('source', 'unknown')}]\n{c['text']}"
            for c in context_chunks
        ])
        
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the context is not enough, say so honestly. Do not invent information.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        resp = co.chat(
            model="command-r-plus",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.message.content[0].text
    
    # ── MAIN METHOD ────────────────────────────────────────────────
    
    def ask(self, query: str, search_k: int = 10, final_k: int = 3) -> dict:
        """
        Full RAG pipeline:
        query → embed → search → rerank → generate
        """
        print(f"\n{'='*50}")
        print(f"Question: {query}")
        print(f"{'='*50}")
        
        # Step 1: Semantic search
        candidates = self.search(query, top_k=search_k)
        print(f"\nFound {len(candidates)} candidates from semantic search")
        
        # Step 2: Rerank
        top_chunks = self.rerank(query, candidates, top_n=final_k)
        print(f"Reranked to top {len(top_chunks)} chunks")
        
        # Step 3: Generate answer
        print("Generating answer...")
        answer_text = self.answer(query, top_chunks)
        
        return {
            "question": query,
            "answer": answer_text,
            "sources_used": [c["text"][:100] + "..." for c in top_chunks],
            "top_rerank_score": top_chunks[0]["rerank_score"] if top_chunks else 0
        }


# ════════════════════════════════════════════════════════════════
# DEMO — use the full system
# ════════════════════════════════════════════════════════════════

rag = CohereRAG(collection_name="cohere_demo")

# Add some documents
rag.add_document(
    text="""
    Cohere is an artificial intelligence company founded in 2019. 
    The company was founded by Aidan Gomez, Ivan Zhang, and Nick Frosst.
    All three founders previously worked at Google Brain, the AI research division of Google.
    Aidan Gomez was also one of the co-authors of the "Attention is All You Need" paper,
    which introduced the Transformer architecture — the foundation of all modern LLMs.
    
    Cohere is focused on enterprise AI. Unlike OpenAI and Anthropic which focus on consumer 
    products, Cohere targets businesses that need private, secure, and customizable AI systems.
    Cohere offers private deployment, which means companies can run Cohere models entirely
    on their own servers. This means sensitive company data never leaves their infrastructure.
    
    Cohere's main product lines are: Command (text generation), Embed (semantic vectors),
    and Rerank (search result re-ordering). These three tools together form a complete
    pipeline for building RAG (Retrieval-Augmented Generation) applications.
    """,
    source_name="cohere_overview"
)

rag.add_document(
    text="""
    Cohere's pricing model is usage-based. You pay for what you use.
    The Command R model costs $0.15 per million input tokens and $0.60 per million output tokens.
    The Command R+ model costs $2.50 per million input tokens and $10.00 per million output tokens.
    The Command R7B model is the cheapest at $0.0375 per million input tokens.
    
    Cohere offers a free trial API key with 1,000 API calls per month.
    The trial key gives access to all models but with rate limits.
    The free tier is suitable for learning and building prototypes.
    For production applications, a paid production key is required.
    
    Cohere's Embed v4 model costs $0.12 per million tokens for text embedding.
    Embed v4 also supports images, making it a multimodal embedding model.
    """,
    source_name="cohere_pricing"
)

# Ask questions
result1 = rag.ask("Who founded Cohere and what is their background?")
print(f"\nAnswer: {result1['answer']}")

result2 = rag.ask("How much does Command R cost?")
print(f"\nAnswer: {result2['answer']}")

result3 = rag.ask("Is Cohere good for personal creative writing projects?")
print(f"\nAnswer: {result3['answer']}")
```

---

## 11. RAG with Cohere's Built-in Connector <a name="11-connector"></a>

### 📖 Theory

Cohere also has a built-in way to do RAG without setting up your own vector database. You can pass documents directly to the `chat()` call using the `documents` parameter. Cohere handles the retrieval internally.

This is great for **quick prototypes** or when your documents are small enough to fit in memory.

### 💻 Code — Inline RAG with Documents Parameter

```python
import cohere
import os

co = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY"))

# Your documents — can be strings or dicts with metadata
documents = [
    {"title": "Cohere Overview", "text": "Cohere was founded in 2019 by Aidan Gomez and team."},
    {"title": "Cohere Products", "text": "Cohere has three main tools: Embed, Rerank, and Command."},
    {"title": "Cohere Pricing", "text": "Command R costs $0.15 per million tokens input."},
    {"title": "Private Deployment", "text": "Cohere allows companies to run models on their own servers."},
    {"title": "Python History", "text": "Python was created by Guido van Rossum in 1991."},
]

response = co.chat(
    model="command-r-plus",
    messages=[{
        "role": "user",
        "content": "What are Cohere's three main tools?"
    }],
    documents=documents   # Pass documents directly — Cohere does the retrieval!
)

print("Answer:")
print(response.message.content[0].text)

# Cohere also tells you which documents it used
if hasattr(response, 'citations') and response.citations:
    print("\nCitations (which documents the answer is based on):")
    for citation in response.citations:
        print(f"  - '{citation.text}' from document index {citation.document_ids}")
```

### When to use this vs. your own vector database?

| | Built-in `documents` param | Your own vector DB |
|---|---|---|
| **Documents** | Small (< 100 documents) | Large (thousands to millions) |
| **Setup** | Zero — just pass them in | Requires ChromaDB / Pinecone setup |
| **Speed** | Slower (re-reads all docs each time) | Fast (pre-indexed) |
| **Cost** | Higher (sends all docs every API call) | Lower (only sends top chunks) |
| **Best for** | Quick prototypes, demos | Production systems |

---

## 12. Common RAG Mistakes and How to Fix Them <a name="12-mistakes"></a>

### ❌ Mistake 1 — Chunks Too Large

**Problem:** You embed entire pages or very long paragraphs. The vector becomes a blurry average of many topics. Search quality drops.

**Fix:** Keep chunks between 200-500 words. Use overlap.

---

### ❌ Mistake 2 — Wrong `input_type`

**Problem:** You use `input_type="search_document"` for both documents AND queries.

**Fix:** Always use `"search_document"` when embedding documents to store, and `"search_query"` when embedding user questions. They are trained differently and must match.

---

### ❌ Mistake 3 — Not Using Reranking

**Problem:** You get top 3 from embedding search and send directly to the LLM. The 3rd result might actually be the most relevant but ranked lower by the embedding model.

**Fix:** Always get top 10-20 from embedding search, then rerank to get the best 3.

---

### ❌ Mistake 4 — Not Telling the LLM to Stay Grounded

**Problem:** You give context but don't instruct the model to ONLY use that context. The model mixes its training knowledge with your documents. You get hallucinations.

**Fix:** Always include in your prompt: *"Answer ONLY based on the context below. If the context is not sufficient, say so."*

---

### ❌ Mistake 5 — No Metadata

**Problem:** You store chunks but no metadata (source file, page number, date). When the model gives a wrong answer, you can't trace where it came from.

**Fix:** Always store at least: `{"source": filename, "chunk_index": i, "date": "..."}`. This lets you debug problems and show users citations.

---

### ❌ Mistake 6 — Re-embedding the Same Documents Every Run

**Problem:** Every time you restart your script, you re-embed all your documents. This wastes API calls and money.

**Fix:** Use `PersistentClient` in ChromaDB (as shown above). Check if documents are already indexed before re-adding them.

```python
# Check before adding
existing = collection.get(ids=["doc_chunk_0"])
if not existing["ids"]:
    rag.add_document(text, source_name)
else:
    print("Document already indexed, skipping.")
```

---

## 13. Critical Thinking — Advanced Questions <a name="13-critical"></a>

### 🤔 Question 1: What is the difference between RAG and Fine-tuning?

| | RAG | Fine-tuning |
|---|---|---|
| **What changes** | The data the model sees at runtime | The model's weights (parameters) |
| **Data stays current** | ✅ Yes — update DB anytime | ❌ No — need to retrain |
| **Cost to update** | Very low | Very high |
| **Model learns style/behavior** | ❌ No | ✅ Yes |
| **Prevents hallucination** | ✅ Strong | ⚠️ Partial |
| **Best for** | Facts, documents, search | Domain behavior, tone, format |

**Rule of thumb:** Use RAG for "what does my document say?" Use fine-tuning for "how should the model behave/sound?"

---

### 🤔 Question 2: How many chunks should I retrieve?

There is a trade-off:

- **Too few (e.g. 1-2):** You might miss the right chunk. Low recall.
- **Too many (e.g. 50):** You give the LLM too much text. It gets confused. Higher cost.
- **Sweet spot:** Retrieve 10-20 with embedding search, rerank to 3-5 for the LLM.

---

### 🤔 Question 3: What if a question needs information from MULTIPLE chunks?

This is called **multi-hop retrieval**. For example:

- "Who founded Cohere and what university did they go to?"
- Chunk 1 has the founder's name
- Chunk 2 has the university

Basic RAG might get only one chunk. Advanced RAG techniques:
- **Multi-query RAG:** Break the question into sub-questions, search for each
- **HyDE (Hypothetical Document Embedding):** Generate a fake answer first, embed it, use it to search

These are advanced topics — but now you know they exist.

---

### 🤔 Question 4: What if the user's question is vague?

Example: "Tell me more" — more about what?

Your RAG system needs to handle conversation history. The solution: **query rewriting**.

```python
def rewrite_query_with_history(history: list, current_query: str) -> str:
    """
    Use an LLM to rewrite a vague query into a specific, self-contained question
    using the conversation history as context.
    """
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
    
    prompt = f"""Given this conversation history:
{history_text}

The user said: "{current_query}"

Rewrite their message as a complete, specific, self-contained search query.
Return ONLY the rewritten query, nothing else."""
    
    response = co.chat(
        model="command-r",   # Cheap model is fine for this task
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content[0].text.strip()

# Example
history = [
    {"role": "user", "content": "Tell me about Cohere's embedding model"},
    {"role": "assistant", "content": "Cohere's Embed v4 model is multimodal..."}
]
vague_query = "How much does it cost?"

specific_query = rewrite_query_with_history(history, vague_query)
print(specific_query)
# Output: "What is the price of Cohere's Embed v4 embedding model?"
```

---

## 🎯 Summary — What You Learned

| Step | Tool | Purpose |
|---|---|---|
| Chunking | Python | Split documents into searchable pieces |
| Embedding (index) | `co.embed` + `input_type="search_document"` | Turn chunks into vectors |
| Storage | ChromaDB / Pinecone | Store and index vectors efficiently |
| Embedding (query) | `co.embed` + `input_type="search_query"` | Turn the question into a vector |
| Search | ChromaDB `.query()` | Find the most similar chunks |
| Reranking | `co.rerank` | Pick the truly most relevant chunks |
| Generation | `co.chat` + context prompt | Generate a grounded, honest answer |

### The golden rule of RAG:

> **"Give the model the right information, and tell it to use only that information."**

If you follow this, your AI system will be accurate, trustworthy, and explainable.

---

## 🔗 Resources

- **Cohere Docs on Embeddings**: https://docs.cohere.com/docs/embeddings
- **Cohere Docs on Reranking**: https://docs.cohere.com/docs/reranking
- **Cohere Docs on RAG**: https://docs.cohere.com/docs/retrieval-augmented-generation-rag
- **ChromaDB Docs**: https://docs.trychroma.com
- **Cohere Playground**: https://dashboard.cohere.com/playground

---

*Deep RAG Tutorial — Built for AI Engineering Students. Take your time with each step. 🚀*
