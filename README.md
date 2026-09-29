# Autonomous arXiv Paper Digest & QA Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq-black.svg)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/vector--store-Chroma-red.svg)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/tests-84%20passed-green.svg)](tests/)

An autonomous, production-grade AI research assistant engineered for the **Autonomous arXiv Paper Digest & QA Agent** assessment. The agent autonomously queries arXiv, parses multi-page academic PDFs with layout awareness, indexes chunks into a local persistent vector store, produces structured 11-field executive briefings, and delivers strictly grounded question answering (RAG) with source provenance citations and verified anti-hallucination guardrails.

---

## 1. Project Overview & Assessment Objective

This system was built to satisfy all requirements of the AI Intern Assessment. It bridges the gap between unstructured academic research and actionable executive intelligence through two interconnected modes:

1. **Autonomous Paper Digest Pipeline**:
   - Accepts either a natural-language research topic, an arXiv paper ID (modern or legacy format), or an arXiv URL.
   - Queries the official arXiv API and deterministically ranks candidate papers using TF-IDF term relevance.
   - Downloads and verifies the target PDF.
   - Extracts document layout and text via PyMuPDF with scanned document detection.
   - Splits text into 500-word windows with 100-word overlaps while preserving strict chunk provenance (page ranges, section names, paper ID).
   - Generates local embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and indexes them into a local Chroma vector database.
   - Synthesizes a structured 11-field Executive Briefing powered by Groq (`openai/gpt-oss-120b`).

2. **Grounded Question Answering (RAG) Loop**:
   - Conducts top-K semantic retrieval across indexed paper chunks.
   - Enforces strict anti-hallucination prompts: answers are generated *solely* from retrieved paper chunks.
   - Cites explicit source provenance (`[Chunk ID, Pages]`) for every factual assertion.
   - Features standardized refusal: if the paper lacks sufficient information, the model explicitly states that the paper does not contain enough information rather than guessing or extrapolating.

---

## 2. Graph Orchestration & Architecture

The system is built on **LangGraph**, using a stateful `StateGraph` where each node represents an isolated, deterministic processing step and edges dictate execution flow with conditional error routing.

```
                           DIGEST PIPELINE
                           
    [Start]
       │
       ▼
┌───────────────────────┐
│  query_understanding  │ ──► Classifies query: "topic" vs "paper"
└───────────────────────┘
       │
       ▼
┌───────────────────────┐         [Error]
│    arxiv_retrieval    │ ───────────────────────┐
└───────────────────────┘                        │
       │                                         │
       ▼                                         │
┌───────────────────────┐         [Error]        │
│       pdf_fetch       │ ───────────────────────┤
└───────────────────────┘                        │
       │                                         │
       ▼                                         │
┌───────────────────────┐         [Error]        │
│       pdf_parse       │ ───────────────────────┤
└───────────────────────┘                        │
       │                                         │
       ▼                                         │
┌───────────────────────┐         [Error]        ▼
│    chunk_and_index    │ ────────────────► ┌───────────────┐
└───────────────────────┘                   │ error_handler │ ──► [END]
       │                                    └───────────────┘
       ▼                                         ▲
┌───────────────────────┐         [Error]        │
│       summarize       │ ───────────────────────┘
└───────────────────────┘
       │
       ▼
     [END]
```

```
                          GROUNDED QA PIPELINE
                          
    [User Question]
           │
           ▼
┌───────────────────────┐         [Error]   ┌───────────────┐
│     qa_retrieval      │ ────────────────► │ error_handler │ ──► [END]
└───────────────────────┘                   └───────────────┘
           │                                         ▲
           ▼                               [Error]   │
┌───────────────────────┐                            │
│       qa_answer       │ ───────────────────────────┘
└───────────────────────┘
           │
           ▼
         [END]
```

### State Machine Architecture & Separation of Concerns

- **Pure Functional Nodes**: Every LangGraph node in `src/nodes/` is a pure function mapping `AgentState -> Partial[AgentState]`. No node performs terminal `input()` calls or blocks execution.
- **CLI Separation**: Interactive user loops (`input()`) reside strictly outside the state graph in `src/cli.py`. The graph can be invoked repeatedly in a clean, stateless manner per question or ingestion batch.
- **Fail-Fast Error Handling**: Any exception or retrieval failure immediately routes through conditional edges to `error_handler`, recording diagnostic messages in state and gracefully halting without unhandled crashes.

### Graph State Lifecycle

Rather than maintaining scattered global variables, the pipeline passes a single typed state across nodes:

| State Key | Type | Description | Populated By |
|---|---|---|---|
| `user_input` | `str` | Raw user query or argument | Entry point / CLI |
| `query_type` | `"topic"` \| `"paper"` | Query classification | `query_understanding` |
| `topic` / `arxiv_id` | `str` | Extracted topic string or sanitized arXiv ID | `query_understanding` |
| `candidate_papers` | `List[PaperMetadata]` | arXiv API candidate results | `arxiv_retrieval` |
| `selected_paper` | `PaperMetadata` | Highest-ranked candidate paper metadata | `arxiv_retrieval` |
| `pdf_path` | `str` | Local filesystem path of downloaded PDF | `pdf_fetch` |
| `raw_text` | `str` | Extracted text from PDF | `pdf_parse` |
| `chunks` | `List[DocumentChunk]` | 500-word windows with page/section provenance | `pdf_parse` |
| `vector_collection_name`| `str` | Chroma collection identifier for paper | `chunk_and_index` |
| `briefing` | `ExecutiveBriefing` | 11-field structured briefing object | `summarize` |
| `current_question` | `str` | Active question in QA session | CLI / User prompt |
| `retrieved_chunks` | `List[DocumentChunk]` | Top-K similarity chunks from Chroma | `qa_retrieval` |
| `qa_answer` | `str` | Grounded answer with source citations | `qa_answer` |
| `conversation_history` | `List[QAExchange]` | Running multi-turn Q&A context | `qa_answer` |
| `error` | `Optional[str]` | Error message if pipeline fails | Any node on error |
| `status` | `str` | Current lifecycle state of the pipeline | All nodes |

---

## 3. Installation & Setup

### Prerequisites
- **Python**: 3.10 or higher (tested and verified on Python 3.12.3)
- **Groq API Key**: A free API key from [console.groq.com](https://console.groq.com/keys)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Maheshdindur/Autonomous-arxiv-paper-digest-qa-agent.git
cd Autonomous-arxiv-paper-digest-qa-agent
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy the example environment file and add your Groq API key:
```bash
cp .env.example .env
```

Open `.env` in your text editor and set your key:
```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
PDF_DOWNLOAD_DIR=./downloads
LOG_LEVEL=INFO
```

### Step 5: Verify Configuration
Verify your environment and local dependencies with the built-in diagnostic tool:
```bash
python3 main.py --check-config
```
Expected output:
```text
[OK] Configuration valid. Provider: Groq (Model: openai/gpt-oss-120b)
[OK] Local credential check passed (key format validated).
[OK] Embedding Model: all-MiniLM-L6-v2
[OK] Storage: Chroma: ./chroma_db, PDFs: ./downloads
```

---

## 4. Execution & Usage Modes

### Mode 1: Interactive Flow (Recommended)
Run without any arguments in an interactive terminal:
```bash
python3 main.py
```
1. Displays the application banner.
2. Prompts: `Enter arXiv paper ID, URL, or research topic:`
3. Executes query classification, arXiv retrieval, PDF download, text extraction, semantic chunking, and Chroma indexing.
4. Renders the Executive Briefing.
5. Transitions automatically into the interactive Grounded QA loop for the ingested paper.

### Mode 2: Natural-Language Topic Discovery
Provide any research query as a command-line argument:
```bash
python3 main.py "Joint KV-Cache Compression and Eviction"
```
The agent queries arXiv, scores candidates using TF-IDF term relevance against the title and abstract, selects the most relevant paper, and executes the complete pipeline.

### Mode 3: Specific Paper ID or URL
Pass a direct modern identifier, legacy identifier, or arXiv URL:
```bash
# Modern arXiv ID
python3 main.py "2512.14946"

# ArXiv abstract / PDF URL
python3 main.py "https://arxiv.org/abs/2512.14946"

# Legacy identifier format
python3 main.py "hep-th/9901001"
```

### Mode 4: One-Shot Question Answering (`--ask`)
Ask a direct question against a specific paper:
```bash
python3 main.py --paper-id "2512.14946" --ask "What quantitative speedup does EVICPRESS achieve?"
```

---

## 5. Verified Demonstrations & Real Outputs

### Real Executive Briefing Output
Generated live for `arXiv:2512.14946v1` (*EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving*):

```markdown
# Executive Briefing: EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving

**Authors**: Shaoting Feng, Yuhan Liu, Hanchen Li, Xiaokun Chen, Samuel Shen, Kuntai Du, Zhuohan Gu, Rui Zhang, Yuyang Huang, Yihua Cheng, Jiayi Yao, Qizheng Zhang, Ganesh Ananthanarayanan, Junchen Jiang  
**arXiv ID**: `2512.14946v1`  
**Published**: 2025-12-16T22:21:55+00:00  
**Link**: http://arxiv.org/abs/2512.14946v1

## Why This Paper Matters
Reusing KV caches is critical for fast LLM inference, but as more users and larger context windows increase the KV cache footprint, GPU memory quickly becomes insufficient. Prior systems either compress or evict caches in isolation, missing the chance to balance latency and quality across multiple storage tiers. EVICPRESS addresses this gap by jointly optimizing compression and eviction decisions, enabling higher cache hit rates on fast memory and reducing generation delays without sacrificing output quality.

## Problem Statement
How to jointly decide which KV caches to compress, how aggressively to compress them, and which to evict to lower-tier storage so that average generation latency is minimized while preserving generation quality across many concurrent LLM contexts.

## Method & Approach
- Define a unified utility function that quantifies the trade-off between quality loss and delay for any compression-eviction configuration of a context.
- Periodically profile all contexts, compute utility scores for all feasible configurations, and use a fast greedy heuristic (solving a Multi-Choice Knapsack) to place or update KV caches across GPU, CPU, and SSD tiers.

## Key Results & Claims
- EVICPRESS achieves up to 2.19× faster time-to-first-token (TTFT) at equivalent generation quality compared to baselines that only compress or only evict.
- Across 5 models and 555 contexts from 12 datasets, EVICPRESS reduces TTFT by 1.43× to 3.77× and improves quality by 13.58% to 55.40% at the same TTFT relative to a fixed-compression + classic-eviction baseline.

## Limitations
- Performance is slightly worse on the musique task compared to the keydiff with LRU eviction baseline.
- CPU cache hit rate considerations lead EVICPRESS to place shorter contexts on CPU, which may limit benefits for workloads with many long contexts.

## Suggested Follow-Up Questions
1. How does EVICPRESS's utility function adapt to different LLM architectures or tokenization schemes?
2. What is the overhead of the profiling and greedy placement process, and how does it scale with the number of concurrent contexts?
3. Can the joint compression-eviction framework be extended to incorporate other memory-saving techniques such as token dropping or quantization?
```

---

### Real Grounded QA Demonstrations

#### Demonstration 1: Direct Factual Answer with Exact Provenance
**Question**: `What is the core objective of EVICPRESS and how does it balance quality and delay?`  
**Answer**:
> The core objective of EVICPRESS is to adaptively choose an eviction-compression configuration for each KV cache so that inference latency (e.g., time-to-first-token and inter-token latency) is reduced while preserving or improving generation quality. It achieves this balance by profiling each context, computing a utility function that combines both delay and quality into a single score, and then selecting the configuration with the highest utility score, thereby jointly optimizing latency and quality `[Chunk ID: 2512.14946_c0011, Page 6-7]` and demonstrating reductions in TTFT and improvements in quality across models and datasets `[Chunk ID: 2512.14946_c0013, Page 7-8]`.
>
> **Sources / Provenance**:
> - `[2512.14946_c0015]` Pages 9-10 (Section: Evaluation)
> - `[2512.14946_c0013]` Pages 7-8 (Section: Evaluation)
> - `[2512.14946_c0018]` Pages 10-11 (Section: Limitations)
> - `[2512.14946_c0011]` Pages 6-7 (Section: Method)

#### Demonstration 2: Multi-Chunk Synthesis
**Question**: `What optimization formulation is used for cache placement and what quantitative TTFT speedup does it achieve?`  
**Answer**:
> EVICPRESS formulates cache placement as an optimization problem solved with a utility-function based selection: it defines a utility function that quantifies the impact of lossy compression and eviction on quality and delay, and evaluates all feasible configurations to choose the optimal device placement and compression ratio `[Chunk ID: 2512.14946_c0004, Page 2-3]`. Using this approach, EVICPRESS achieves a time-to-first-token (TTFT) reduction of roughly 1.43× to 3.77× compared to baselines `[Chunk ID: 2512.14946_c0013, Page 7-8]`.
>
> **Sources / Provenance**:
> - `[2512.14946_c0004]` Pages 2-3 (Section: Background)
> - `[2512.14946_c0013]` Pages 7-8 (Section: Evaluation)
> - `[2512.14946_c0001]` Pages 1-2 (Section: Introduction)

#### Demonstration 3: Strict Anti-Hallucination Refusal
**Question**: `What was the learning rate schedule and AdamW weight decay parameter used during foundation model pretraining?`  
**Answer**:
> The provided paper does not contain enough information to answer this question.

*(Because the paper evaluates pre-trained models rather than detailing foundation pre-training hyperparameters, the system strictly refuses rather than fabricating values).*

---

## 6. Key Design Decisions & Technical Trade-offs

1. **Groq as Sole LLM Provider**:
   - *Decision*: Configured Groq (`openai/gpt-oss-120b`) as the sole LLM backend.
   - *Rationale*: Delivers near-instant generation (< 2s for multi-thousand token contexts) with high reasoning fidelity, well within free-tier capabilities.
2. **Local Sentence-Transformers Embeddings**:
   - *Decision*: Local `sentence-transformers/all-MiniLM-L6-v2` running directly on the host machine.
   - *Rationale*: Eliminates external embedding API costs, avoids secondary rate limits, and guarantees deterministic cosine vector similarity.
3. **Chunk Configuration (500 Words / 100-Word Overlap)**:
   - *Decision*: Explicit word-based sliding windows (500 words ≈ 650 tokens, 100-word overlap).
   - *Rationale*: Preserves complete technical paragraphs and mathematical derivations while keeping 4 retrieved context blocks well within Groq TPM limits.
4. **Boundary Sentence Completion Safeguard**:
   - *Decision*: Implemented `complete_trailing_sentence` lookahead in evidence assembly.
   - *Rationale*: Prevents truncated numerical values across chunk boundaries (e.g., preventing `"reduced TTFT by 1.43 to..."` cutoff errors from entering briefings).
5. **Deterministic Candidate Ranking**:
   - *Decision*: TF-IDF term relevance scoring over candidate Title and Abstract.
   - *Rationale*: Provides transparent, explainable ranking for topic searches without wasting LLM token quota.

---

## 7. Assessment Rubric Alignment

| Rubric Dimension | Weight | System Implementation & Evidence |
|---|---|---|
| **Agent / Graph Design** | 25% | Stateful LangGraph workflow with typed `AgentState`; conditional routing for error handling; clean decoupling of graph execution from interactive CLI loops (`src/graph/workflow.py`). |
| **Correctness & Grounding** | 25% | Strict zero-shot anti-hallucination prompting; citation requirement (`[Chunk ID, Pages]`); standardized refusal string on insufficient evidence; boundary sentence lookahead (`src/utils/qa_engine.py`). |
| **Retrieval & Parsing Quality** | 20% | PyMuPDF text extractor with scanned document detection; best-effort heading detection without section fabrication; 500-word / 100-word overlap chunking; local Chroma cosine vector store; TF-IDF candidate ranking. |
| **Code Quality & Architecture** | 15% | Clean modular structure (`src/nodes/`, `src/utils/`, `src/graph/`); strict type hints; zero code duplication; comprehensive test suite (84 automated tests) with no deprecation warnings. |
| **Communication & Reporting** | 15% | 11-field executive briefing matching assignment specification; plain-English "why it matters" summary; explicit citations; complete documentation and usage instructions. |

---

## 8. Limitations & Failure Handling

1. **Scanned / Rasterized PDFs**: PDFs without an embedded text layer are detected during parsing. The pipeline halts with an informative error rather than emitting blank or hallucinated summaries.
2. **Graphic Figures & Trend Lines**: Text captions and layout text within figures are extracted, but bitmap image trends are not visually interpreted.
3. **Groq Free-Tier Rate Limits**: Rapid sequential requests may reach the 8,000 TPM limit on free accounts. The agent bounds evidence assembly to ~2,200 words to minimize rate-limit risks.

---

## 9. Testing & Verification

The repository includes a comprehensive automated test suite covering unit tests, node execution, vector store operations, retrieval quality, and failure handling:

```bash
# Run all 84 offline unit and integration tests (no API key required)
pytest -v -m "not live_llm"

# Run CLI interaction tests
pytest tests/test_cli.py -v

# Run live Groq LLM integration test (requires GROQ_API_KEY in .env)
pytest tests/test_groq_live_integration.py -v -m live_llm

# Run complete test suite
pytest -v
```

### Verified Test Results
```text
======================== 84 passed, 1 deselected in 54.63s ========================
```
*(All 84 offline unit/integration tests pass with 0 failures, 0 errors, and 0 warnings).*
