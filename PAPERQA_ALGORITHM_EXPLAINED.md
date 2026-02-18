# PaperQA Algorithm Explanation

## Overview

PaperQA2 is a Retrieval Augmented Generation (RAG) system designed for scientific literature question answering. It achieves superhuman performance by combining document indexing, semantic search, LLM-based re-ranking, and contextual summarization.

## Core Architecture

### 1. Document Storage (`Docs` class)

The system maintains:
- **`docs`**: Dictionary of document metadata (titles, DOIs, citations, authors)
- **`texts`**: List of text chunks extracted from documents
- **`texts_index`**: Vector store for semantic search (default: NumpyVectorStore)
- **`docnames`**: Set of unique document names

### 2. Data Structures

**Text Chunk (`Text`):**
- Contains text content, embeddings, document reference, and optional media (images/tables)
- Each chunk has metadata including page numbers and file location

**Context (`Context`):**
- Represents a relevant passage with a contextual summary
- Includes a relevance score (0-10) from LLM evaluation
- Links back to source text, document, and citation

**Session (`PQASession`):**
- Tracks the question, collected contexts, answer, and token usage
- Maintains conversation state across the workflow

## Three-Phase Algorithm

### Phase 1: Paper Search

**Tool:** `paper_search(query, min_year, max_year)`

**Process:**
1. Agent generates keyword queries from the user question
2. Searches a full-text index (using tantivy library) over local PDF/document collection
3. Returns top k papers (default: 8) matching the query
4. Chunks returned papers into overlapping text segments
5. Embeds chunks using embedding model (default: OpenAI `text-embedding-3-small`)
6. Adds chunks to the vector store

**Key Features:**
- Supports continuation: calling with same query returns next batch
- Metadata enrichment: Automatically fetches citation counts, journal quality, DOIs from Crossref/Semantic Scholar
- Multimodal support: Can extract and embed images/tables from PDFs

### Phase 2: Gather Evidence

**Tool:** `gather_evidence(question)`

**Process:**
1. **Retrieval:** Embed the question and perform Maximum Marginal Relevance (MMR) search
   - Retrieves top `evidence_k` chunks (default: 10) from vector store
   - MMR balances relevance and diversity to avoid redundant passages

2. **Contextual Summarization:** For each retrieved chunk:
   - LLM (`summary_llm`) receives the chunk text and the question
   - Generates a concise summary (~100 words) focused on answering the question
   - **Critically:** LLM assigns a relevance score (0-10)
   - Media (images/tables) are included in the prompt if available

3. **LLM Re-ranking:**
   - Filters out contexts with score ≤ threshold (default: 0)
   - Keeps only relevant and diverse evidence
   - Each context includes the summary, score, and citation

**Innovation:** This is **not** standard RAG. The system creates query-specific summaries rather than using raw chunks, and LLM-based scoring is more accurate than pure similarity metrics.

### Phase 3: Generate Answer

**Tool:** `gen_answer()`

**Process:**
1. **Context Aggregation:**
   - Selects top contexts (up to `answer_max_sources`, default: 5)
   - Serializes contexts with their summaries and citations
   - Groups by question if multiple gather_evidence calls were made

2. **Answer Generation:**
   - Prompts the main LLM with:
     - The original question
     - Aggregated context passages
     - Citation format instructions
   - LLM generates answer with inline citations like `(Author2023 pages 4-5)`

3. **Post-processing:**
   - Extracts citations and builds bibliography
   - Formats final answer with citation list
   - Can optionally run post-prompt for refinement

**Special Handling:**
- If no contexts available: Returns "I cannot answer..."
- Filters out background information not directly relevant
- Supports iterative answering (agent can call gen_answer multiple times)

## Key Design Decisions

### 1. Chunking Strategy
- Overlapping chunks (typically ~3000 chars with overlap)
- Preserves context across chunk boundaries
- Different readers available: PyPDF, PyMuPDF, Docling, Nemotron

### 2. Embedding with Enrichment
- Standard dense embeddings (OpenAI/local via sentence-transformers)
- Hybrid embeddings: combines dense + sparse (keyword-based)
- **Media enrichment:** LLM generates synthetic captions for images/tables to improve retrieval

### 3. MMR Search
- Prevents redundant evidence by maximizing diversity
- Lambda parameter (default: 1.0) controls relevance vs. diversity tradeoff
- Formula: `MMR = λ·similarity(query, doc) - (1-λ)·max_similarity(doc, selected_docs)`

### 4. Contextual Summarization (RCS)
- Each chunk summarized in context of the specific question
- More accurate than raw semantic similarity
- Summaries are concise and focused
- JSON parsing ensures structured output with summary + score

### 5. LLM-Based Scoring
- 0-10 relevance scale
- More nuanced than cosine similarity
- Enables filtering of tangentially related content

### 6. Metadata-Aware System
- Automatically enriches papers with citation counts, journal impact
- Checks for retractions
- Provides formatted citations in multiple styles

## Agentic Workflow

PaperQA2 uses an LLM agent (`ToolSelector`) that:
- Decides which tools to invoke and in what order
- Can call tools in parallel when safe
- Typical pattern: `paper_search` → `gather_evidence` → `gen_answer`
- Can iterate: refine search, gather more evidence, regenerate answer
- Agent maintains state and tracks costs

**Tool Safety:**
- `paper_search`: Concurrency-safe (independent operations)
- `gather_evidence`: Not parallel with itself (modifies question state)
- `gen_answer`: Not parallel with itself (generates final answer)

## Configuration

The system is highly configurable via `Settings`:
- LLM models for different tasks (agent, summary, answer)
- Embedding models (OpenAI, local, hybrid)
- Number of evidence pieces (`evidence_k`)
- Number of final sources (`answer_max_sources`)
- Answer length, summary length
- Rate limits and cost tracking

## Performance Optimizations

1. **Caching:** Embeddings cached when `Docs` object is pickled
2. **Concurrency:** Parallel context creation (default: 4 concurrent)
3. **Lazy embedding:** Option to defer embedding until summarization
4. **Efficient search:** Full-text index before vector search reduces candidates
5. **Batch processing:** Multiple LLM calls batched when possible

## Novel Contributions

1. **Contextual Summarization with Re-ranking:** Unlike standard RAG which uses raw chunks
2. **Agentic tool use:** Agent decides search strategy dynamically
3. **Metadata integration:** Rich paper metadata improves quality
4. **Multimodal support:** Images/tables enhance retrieval and answering
5. **Scientific focus:** Optimized for academic literature with proper citations

## Comparison to Standard RAG

| Aspect | Standard RAG | PaperQA2 |
|--------|--------------|----------|
| Retrieval | Raw chunks | Contextual summaries |
| Ranking | Cosine similarity | LLM-based scoring |
| Workflow | Fixed pipeline | Agentic/iterative |
| Citations | Often missing | Inline with sources |
| Metadata | Limited | Rich (citations, impact, DOI) |
| Multimodal | Rare | Images/tables supported |

## Codebase Structure

- `docs.py`: Core `Docs` class with add/query methods
- `agents/tools.py`: Three main tools (search, gather, answer)
- `agents/env.py`: Agent environment and state management
- `core.py`: LLM parsing and context creation logic
- `llms.py`: Vector stores and embedding models
- `types.py`: Data models (Doc, Text, Context, PQASession)
- `prompts.py`: Prompt templates for all LLM calls
- `readers.py`: PDF/document parsing

## Example Flow

```
User Question: "How do transformers handle long sequences?"

1. Agent calls paper_search("transformer long sequence attention")
   → Finds 8 papers, chunks them, embeds chunks

2. Agent calls gather_evidence("How do transformers handle long sequences?")
   → Retrieves top 10 chunks via MMR
   → LLM summarizes each: "This passage describes sparse attention..."
   → Scores: [8, 7, 9, 3, 7, 2, 8, 6, 4, 8]
   → Filters out scores ≤ 0, keeps 10 contexts

3. Agent calls gen_answer()
   → Selects top 5 contexts by score
   → Generates: "Transformers handle long sequences through sparse attention
      mechanisms (Vaswani2017 pages 3-4) and sliding window approaches
      (Beltagy2020 pages 2-3)..."

4. Returns formatted answer with bibliography
```

## Summary

PaperQA2's algorithm combines:
- Full-text + semantic search for comprehensive retrieval
- LLM-powered contextual summarization for accuracy
- Intelligent re-ranking for relevance
- Agentic workflow for flexibility
- Rich metadata for scientific credibility

The key innovation is **contextual summarization**: instead of feeding raw chunks to the answer LLM, each chunk is first processed by a summary LLM that extracts only information relevant to the specific question and scores its relevance. This dramatically improves answer quality over traditional RAG systems.
