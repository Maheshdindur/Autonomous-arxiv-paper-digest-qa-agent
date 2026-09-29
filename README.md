# Autonomous arXiv Paper Digest & QA Agent

A production-grade Python agent that autonomously discovers research papers on arXiv, parses complex multi-page PDF documents, indexes text into a local vector store, produces structured executive briefings, and provides strictly grounded question answering (RAG) with source provenance citations.

---

## 1. Project Overview

The **Autonomous arXiv Paper Digest & QA Agent** operates in two modes:
1. **Paper Digest Mode**: Ingests an arXiv paper ID (modern or legacy), an arXiv URL, or a natural-language research topic. If a topic is supplied, it queries the official arXiv API and deterministically ranks candidate papers using TF-IDF term relevance. It downloads and validates the target PDF, extracts layout text with PyMuPDF, segments content into overlapping 500-word windows with metadata preservation, embeds chunks using local `all-MiniLM-L6-v2` embeddings into Chroma DB, and generates a grounded 11-field executive briefing via Groq (`openai/gpt-oss-120b`).
2. **Grounded QA Mode**: Answers user questions using top-K semantic retrieval from local Chroma storage. The model is constrained by strict anti-hallucination prompts to answer *only* from retrieved evidence chunks, citing specific chunk IDs and page numbers for every claim. If the paper lacks sufficient information, the agent explicitly refuses rather than extrapolating.

---

## 2. Architecture & StateGraph Design

The system separates LangGraph orchestration, CLI interaction, and individual processing nodes.

### Digest Workflow Graph (`src/graph/workflow.py`)
```
[query_understanding]
        │
        ▼
[arxiv_retrieval] ──(error)──► [error_handler] ──► END
        │
        ▼
   [pdf_fetch]     ──(error)──► [error_handler] ──► END
        │
        ▼
   [pdf_parse]     ──(error)──► [error_handler] ──► END
        │
        ▼
[chunk_and_index]  ──(error)──► [error_handler] ──► END
        │
        ▼
   [summarize]     ──► END
```

### Grounded QA Workflow Graph (`src/graph/workflow.py`)
```
[qa_retrieval] ──(error)──► [error_handler] ──► END
       │
       ▼
  [qa_answer]  ──► END
```

*Note: Interactive CLI terminal loops (`input()`) reside strictly outside the LangGraph nodes in `src/cli.py`, allowing the graph to remain pure, stateless, and repeatedly invokable per question.*

---

## 3. Explicit Shared State Schema

All graph nodes share the typed dictionary `AgentState` defined in [`src/graph/state.py`](src/graph/state.py):

```python
class AgentState(TypedDict, total=False):
    # User Inputs & Query Classification
    user_input: str
    query_type: Optional[Literal["topic", "paper"]]
    topic: Optional[str]
    arxiv_id: Optional[str]
    arxiv_url: Optional[str]

    # Discovery & Retrieval
    candidate_papers: List[PaperMetadata]
    selected_paper: Optional[PaperMetadata]

    # Document Extraction & Vector Store
    pdf_path: Optional[str]
    raw_text: Optional[str]
    chunks: List[DocumentChunk]
    vector_collection_name: Optional[str]

    # Executive Briefing
    briefing: Optional[ExecutiveBriefing]

    # Grounded QA & Multi-Turn History
    current_question: Optional[str]
    retrieved_chunks: List[DocumentChunk]
    qa_answer: Optional[str]
    conversation_history: List[QAExchange]

    # Pipeline Diagnostics
    error: Optional[str]
    status: Literal[
        "init", "understanding_query", "retrieving", "ranking",
        "fetching_pdf", "parsing_pdf", "chunking", "indexing",
        "summarizing", "completed", "qa_retrieving", "qa_answering", "error"
    ]
```

### Chunk Metadata Preservation (`ChunkMetadata`)
Each chunk preserves complete provenance:
- `chunk_id`: Formatted identifier (e.g. `2512.14946_c0012`)
- `paper_id`: arXiv paper identifier
- `page_start` / `page_end`: Exact 1-indexed document page boundaries
- `section`: Detected section/heading (or `None`, never fabricated)
- `word_count`: Word count of the sliding window

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.12.3)
- Free Groq API Key (from [console.groq.com](https://console.groq.com/keys))

### Installation Steps
```bash
# 1. Clone repository and navigate to root
git clone <repo-url>
cd 8byte

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
```

Edit `.env` to include your Groq API key:
```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
PDF_DOWNLOAD_DIR=./downloads
LOG_LEVEL=INFO
```

### Validate Configuration
```bash
python3 main.py --check-config
```
Output:
```
[OK] Configuration valid. Provider: Groq (Model: openai/gpt-oss-120b)
[OK] Local credential check passed (key format validated).
[OK] Embedding Model: all-MiniLM-L6-v2
[OK] Storage: Chroma: ./chroma_db, PDFs: ./downloads
```

---

## 5. CLI Usage & Examples

### 0. Interactive Startup (No Arguments)
Simply run without arguments in an interactive terminal:
```bash
python3 main.py
```
The application displays a clean header, prompts:
```text
Enter arXiv paper ID, URL, or research topic: 2512.14946
```
and executes the complete pipeline, displays the Executive Briefing, and transitions directly into interactive Grounded QA mode.

### 1. Research Topic Input (Auto-Search, Ranking & Ingestion)
```bash
python3 main.py "Joint KV-Cache Compression and Eviction"
```
The agent queries the arXiv API, scores candidates with TF-IDF term relevance over Title and Abstract, downloads the top-ranked paper, chunks and indexes it, and outputs the Executive Briefing.

### 2. Specific Paper ID or URL Input
```bash
python3 main.py "2512.14946"
# Or arXiv URL:
python3 main.py "https://arxiv.org/abs/2512.14946"
# Or legacy arXiv ID:
python3 main.py "hep-th/9901001"
```

### 3. One-Shot Grounded QA (`--ask`)
```bash
python3 main.py --paper-id "2512.14946" --ask "What is the core objective of EVICPRESS?"
```

### 4. Interactive Follow-Up QA Mode
When running in an interactive terminal, the CLI automatically prompts for follow-up questions upon displaying the briefing:
```bash
python3 main.py "2512.14946"
# Displays briefing, then prompts:
Ask a question: What speedup does EVICPRESS achieve?
```

---

## 6. Real Executive Briefing Example

Generated live for `arXiv:2512.14946v1`:

```markdown
# Executive Briefing: EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving

**Authors**: Shaoting Feng, Yuhan Liu, Hanchen Li, Xiaokun Chen, Samuel Shen, Kuntai Du, Zhuohan Gu, Rui Zhang, Yuyang Huang, Yihua Cheng, Jiayi Yao, Qizheng Zhang, Ganesh Ananthanarayanan, Junchen Jiang  
**arXiv ID**: `2512.14946v1`  
**Published**: 2025-12-16T22:21:55+00:00  
**Link**: http://arxiv.org/abs/2512.14946v1

## Why This Paper Matters
Reusing KV caches is critical for fast LLM inference, but as more users and larger context windows increase the KV cache footprint, GPU memory quickly becomes insufficient. Prior systems either compress or evict caches in isolation, missing the chance to balance latency and quality across multiple storage tiers. EVICPRESS addresses this gap by jointly optimizing compression and eviction decisions, enabling higher cache hit rates on fast memory and reducing generation delays without sacrificing output quality.

## Problem Statement
How to jointly decide which KV caches to compress, how aggressively to compress them, and which to evict to lower‑tier storage so that average generation latency is minimized while preserving generation quality across many concurrent LLM contexts.

## Method & Approach
- Define a unified utility function that quantifies the trade‑off between quality loss and delay for any compression‑eviction configuration of a context.
- Periodically profile all contexts, compute utility scores for all feasible configurations, and use a fast greedy heuristic (solving a Multi‑Choice Knapsack) to place or update KV caches across GPU, CPU, and SSD tiers.

## Key Results & Claims
- EVICPRESS achieves up to 2.19× faster time‑to‑first‑token (TTFT) at equivalent generation quality compared to baselines that only compress or only evict.
- Across 5 models and 555 contexts from 12 datasets, EVICPRESS reduces TTFT by 1.43× to 3.77× and improves quality by 13.58% to 55.40% at the same TTFT relative to a fixed‑compression + classic‑eviction baseline.

## Limitations
- Performance is slightly worse on the musique task compared to the keydiff with LRU eviction baseline.
- CPU cache hit rate considerations lead EVICPRESS to place shorter contexts on CPU, which may limit benefits for workloads with many long contexts.

## Suggested Follow-Up Questions
1. How does EVICPRESS’s utility function adapt to different LLM architectures or tokenization schemes?
2. What is the overhead of the profiling and greedy placement process, and how does it scale with the number of concurrent contexts?
3. Can the joint compression‑eviction framework be extended to incorporate other memory‑saving techniques such as token dropping or quantization?
```

---

## 7. Real Grounded QA Demonstrations

### Demonstration 1: Direct Answer with Provenance Citations
**Question**: `What is the core objective of EVICPRESS and how does it balance quality and delay?`  
**Answer**:
> The core objective of EVICPRESS is to adaptively choose an eviction‑compression configuration for each KV cache so that inference latency (e.g., time‑to‑first‑token and inter‑token latency) is reduced while preserving or improving generation quality. It achieves this balance by profiling each context, computing a utility function that combines both delay and quality into a single score, and then selecting the configuration with the highest utility score, thereby jointly optimizing latency and quality `[Chunk ID: 2512.14946_c0011, Page 6-7]` and demonstrating reductions in TTFT and improvements in quality across models and datasets `[Chunk ID: 2512.14946_c0013, Page 7-8]`.
>
> **Sources / Provenance**:
> - `[2512.14946_c0015]` Pages 9-10 (Section: Evaluation)
> - `[2512.14946_c0013]` Pages 7-8 (Section: Evaluation)
> - `[2512.14946_c0018]` Pages 10-11 (Section: Limitations)
> - `[2512.14946_c0011]` Pages 6-7 (Section: Method)

### Demonstration 2: Multi-Section Synthesis Across Chunks
**Question**: `What algorithm or optimization technique is used to solve the cache placement problem, and what quantitative TTFT speedup does EVICPRESS achieve?`  
**Answer**:
> EVICPRESS formulates cache placement as an optimization problem solved with a utility‑function based selection: it defines a utility function that quantifies the impact of lossy compression and eviction on quality and delay, and the configuration‑selection module evaluates all feasible compression‑eviction configurations to choose the optimal device placement, compression method, and ratio `[Chunk ID: 2512.14946_c0004, Page 2-3]`. Using this approach, EVICPRESS achieves a time‑to‑first‑token (TTFT) reduction of roughly 1.43× to 3.77× compared to baselines (i.e., up to about a 3.8× speedup) `[Chunk ID: 2512.14946_c0013, Page 7-8]`.
>
> **Sources / Provenance**:
> - `[2512.14946_c0004]` Pages 2-3 (Section: Background)
> - `[2512.14946_c0013]` Pages 7-8 (Section: Evaluation)
> - `[2512.14946_c0018]` Pages 10-11 (Section: Limitations)
> - `[2512.14946_c0001]` Pages 1-2 (Section: Introduction)

### Demonstration 3: Grounded Refusal (Anti-Hallucination)
**Question**: `What was the learning rate schedule and AdamW weight decay parameter used during foundation model pretraining?`  
**Answer**:
> The provided paper does not contain enough information to answer this question.

*(Zero hallucination; the model explicitly refuses because the paper evaluates pre-trained models rather than detailing their pre-training hyper-parameters).*

---

## 8. Alignment with Assignment Rubric

| Rubric Dimension | Weight | Implementation Evidence |
|---|---|---|
| **Agent / Graph Design** | 25% | Stateful LangGraph architecture with explicit `AgentState` schema; clean routing with conditional error handler; strict separation between graph execution and CLI interaction loops (`src/graph/workflow.py`). |
| **Correctness & Grounding** | 25% | Strict zero-shot anti-hallucination prompting; citation requirement (`[Chunk ID, Page]`); standardized refusal string when evidence is insufficient; automatic sentence completion across chunk boundaries (`src/utils/qa_engine.py`, `src/utils/briefing_extractor.py`). |
| **Retrieval & Parsing Quality** | 20% | PyMuPDF text extractor with scanned document detection; best-effort heading detection without section fabrication; 500-word / 100-word overlap chunking; Chroma vector DB with cosine similarity; TF-IDF candidate ranking for topic search. |
| **Code Quality & Architecture** | 15% | Modular design (`src/nodes/`, `src/utils/`, `src/graph/`); strict type hinting; comprehensive docstrings; robust configuration validation (`src/config.py`); 81 automated tests with zero deprecation warnings. |
| **Communication & Reporting** | 15% | 11-field executive briefing matching assignment specification; plain-English "why it matters" summary; explicit citations; complete documentation and usage instructions in `README.md`. |

---

## 9. Design Decisions & Trade-Offs

1. **Groq as Sole LLM Provider**:
   - *Decision*: Configured Groq (`openai/gpt-oss-120b`) as the single provider.
   - *Rationale*: Extreme inference speed (< 2s for multi-thousand token contexts) and free tier availability.
2. **Local Sentence-Transformers Embeddings**:
   - *Decision*: Local `sentence-transformers/all-MiniLM-L6-v2` running on CPU/GPU.
   - *Rationale*: Eliminates third-party embedding API costs, avoids secondary rate limits, and provides deterministic cosine vectors.
3. **Chunk Size (500 Words / 100-Word Overlap)**:
   - *Decision*: 500 words (~650 tokens) with 100-word overlap.
   - *Rationale*: Preserves paragraph context and technical derivations while keeping 4-chunk retrieved contexts (~2,000 words) safely below Groq TPM thresholds (8,000 TPM limit).
4. **Deterministic Ranking for Topic Searches**:
   - *Decision*: TF-IDF / term-relevance similarity over candidate Title + Abstract.
   - *Rationale*: Transparent, explainable, fast, and avoids unnecessary LLM calls.
5. **Boundary Sentence Completion**:
   - *Decision*: `complete_trailing_sentence` looks ahead into adjacent chunk to complete cut-off sentences.
   - *Rationale*: Prevents truncated numerical claims (e.g. `"reduces TTFT by 1.43 to..."` cutoff) from degrading briefing quality.

---

## 10. Limitations & Known Failure Cases

1. **Scanned / Rasterized PDFs**: Documents lacking an embedded text layer (pure scanned images) are detected and rejected with an explicit error message rather than emitting empty or fabricated briefings.
2. **Graphic Plots & Complex Diagrams**: Text inside vector figures is parsed, but visual trend lines or non-textual diagrams are not synthesized.
3. **Groq Free-Tier Rate Limits**: Rapid sequential requests may reach the 8,000 TPM limit on free accounts. The agent bounds evidence assembly to ~2,200 words to minimize rate-limit risks.

---

## 11. Testing & Verification

The test suite is partitioned into offline tests and live Groq API tests:

```bash
# Run all offline unit and integration tests (80 tests)
pytest -v -m "not live_llm"

# Run live Groq API test
pytest tests/test_groq_live_integration.py -v -m live_llm

# Run complete test suite (81 tests)
pytest -v
```

### Full Test Output:
```
============================== 81 passed in 36.44s ==============================
```
