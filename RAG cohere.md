# Cohere RAG - The complete deep tutorial

## What is RAG - And what is the problem it solves?

First thing before we Know what is the **RAG**, We should know the problem 

Imagine if you are an doctor that have been study for 10 years (*Traning*)

And a new drug have been released, What you will gonna do? you will read book, or search about the new *released drug* and then answer your **patient's**

That is the same thing in the LLM

The 3 problems in every LLM:

Problem | What happens |
|---|---|
Knowledge cut off | When the model have been trained at 2025 so it's knowledge cut off at 2022 and the user asked the model about something happend in 2026 for exmaple: "what is the country that left OPEC" The model will not know |
Hallucination | When you ask the model about something that it does not know sometime it's give you an wrong answers, so when an customer ask's the AI about something about your company, And the model does not know it will Hallucinate and give wrong answers, And RAG reduce the Hallucination significantly 

RAG solve all these problems by saving your data and when needed it will go to search the data and will give the client an answer about the something that **He is asing about** from your documents

## How RAG works

RAG have 2 phases

First one: **Indexing** (*You do it once, or when data changed*)

Your documents -> split into chunks -> Embed each *chunk* -> store vetors in a **database** 

Second one: **Queryring** (*Happens every time the user ask a question*)

User Question -> Embed the question -> Search on it on the DataBase -> Get top chunks -> rerank -> LLM generate answer

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

**Every step does matter, And we will go on each one deeply**

## The 3 Core Cohere Tools in RAG

| Tool | Role in RAG | What it does | 
|---|---|---|
| **Embed** | Embed + Querying | Convert the text into vectors | 
| **Rerank** | Querying | Re-order results by relevance To the query |
| **Command** | Querying | Generates the **Final answer** |

>  Why does Cohere have all 3? Most providers only give you the LLM (like GPT). Cohere was built from the ground up for enterprise search and RAG, so they built all the tools you need in one place. This is a big advantage — you don't need to mix and match from different companies.

# Step 1 for using RAG ~ Chunks

You cannot embed a full book at once as one piece, and here is why:

**Tokens** — Every LLM has a token limit for processing things at once. For example, GPT-4 has a 128k token limit. If your book is 500k tokens, it simply will not fit.

**Search quality drops** — If a chunk is too big, the vector becomes a "blurry" average of many topics. Imagine searching for "how to make coffee" but the chunk also talks about tea, juice, and water. The vector gets confused between all of them, so your search result becomes weak and inaccurate.

**LLM context is expensive** — Putting a full book in the context costs a lot of money. APIs like OpenAI charge per token, so sending 500 pages every single query will drain your budget fast.

So we split our text into pieces, usually **200–500 words** per chunk. And there are three main ways to do this:

---

## Way 1 — Split on every N characters or words (Dumbest way)

N means: *"how many characters should each piece be?"*

**Example:**
```
Input:  "The cats are so cute and fluffy."
N = 10
Output: ['The cats a', 're so cute', ' and fluff', 'y.']
```

Another example with words instead of characters:
```
Input:  "Artificial intelligence is changing the world very fast"
N = 3 words
Output: ['Artificial intelligence is', 'changing the world', 'very fast']
```

**Cons:**
- Cuts words in half — `"New Yo"` / `"rk"` destroys meaning
- Has zero understanding of sentences or topics
- A vector embedding of `"The cats a"` means nothing to a search engine
- You might cut right in the middle of the most important sentence in your document

**When do engineers still use it?** — When processing raw binary data or logs where structure does not matter, or when speed is the only priority.

---

## Way 2 — Cutting on punctuation marks (Better)

Cuts the chunk at every `.` `!` `?` so you always get full, complete sentences.

**Example:**
```
Input: "The sky is blue. Birds can fly! Do they migrate? Yes they do."

Output:
Chunk 1 → "The sky is blue."
Chunk 2 → "Birds can fly!"
Chunk 3 → "Do they migrate?"
Chunk 4 → "Yes they do."
```

Much cleaner. Every chunk is a real, meaningful sentence.

**Cons:**
- Some sentences are extremely short and carry almost no information — `"Yes."` alone is a useless chunk
- Some sentences are extremely long — a 3-line legal sentence becomes one giant chunk
- It ignores **paragraphs and topics** — two sentences about completely different subjects might land in the same chunk just because they are next to each other
- No connection between chunks — if an idea spans two sentences, you lose the link

---

## Way 3 — Overlap chunking (Best for most cases)

Each chunk overlaps the next one by a percentage, usually **20–50%**. This makes sure no idea gets lost between two chunks.

**Example without overlap (the problem):**
```
Chunk 1: "The sky is blue. Birds can fly."
Chunk 2: "They migrate south in winter. Temperatures drop."

Problem: The connection between "birds fly" and "they migrate" is completely lost.
         If someone asks "where do birds go?" — no single chunk has the full answer.
```

**Example with 50% overlap (the fix):**
```
Chunk 1: "The sky is blue. Birds can fly."
Chunk 2: "Birds can fly. They migrate south in winter."
Chunk 3: "They migrate south in winter. Temperatures drop."

Now chunk 2 captures BOTH ideas together. The search finds the full picture.
```

**Real world example — a medical document:**
```
Without overlap:
Chunk 1: "The patient must take 500mg of ibuprofen."
Chunk 2: "This should be taken after meals only."

With overlap:
Chunk 1: "The patient must take 500mg of ibuprofen."
Chunk 2: "Take 500mg of ibuprofen. This should be taken after meals only."

If a doctor's assistant asks "how should the patient take ibuprofen?"
Only the overlap version answers it correctly in one chunk.
```

**Cons:**
- You store **more data** — with 50% overlap, you are roughly doubling your number of chunks
- More chunks = more vectors in your database = **slower search and higher storage cost**
- If overlap is too high (like 90%), almost every chunk is a duplicate of the previous one — completely wasteful

---

## Quick Comparison

| | N-character | Punctuation | Overlap |
|---|---|---|---|
| Breaks words? | Yes | No | No |
| Keeps full sentences? | No | Yes | Yes |
| Keeps context between ideas? | No | No | Yes |
| Storage cost | Low | Low | High |
| Best for | Raw logs | Simple docs | Most RAG apps |

---

## 🔥 Thinking question

In overlap chunking — if you set your overlap to **100%**, what do you think will happen to your RAG system? And what is the **minimum useful overlap percentage** before it becomes pointless, and why?

## NOW LET'S CODEEEEEE!!!!!!

