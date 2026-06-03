# Discharge Summary Agent

An agentic AI system that reads patient source notes (PDFs) and produces structured, clinically safe discharge summary drafts for clinician review.

## Architecture

### Agent Loop Design
The system uses a **CrewAI sequential multi-agent pipeline** with 4 specialized agents:

1. **Orchestrator** — Plans execution, ingests PDFs, coordinates agents
2. **Conflict Detector** — Scans all notes for contradictions, escalates every conflict found
3. **Medication Reconciler** — Diffs admission vs discharge meds, checks drug interactions, flags undocumented changes
4. **Summary Writer** — Assembles the final structured draft using only sourced facts

Each agent has a hard `max_iter` cap (Orchestrator: 20, others: 10) to prevent infinite loops.

### PDF Ingestion
Source PDFs are scanned documents (handwritten + printed). PyMuPDF extracts pages as images, which are sent to **GPT-4o Vision** for text extraction. Extracted text is cached locally to avoid re-processing costs on subsequent runs.

```
data/
└── patient_folder/
├── source.pdf
└── .cache/
└── source.json   ← cached vision extraction
```

## No-Fabrication Guardrail

This is the core safety property of the system. It is enforced at multiple levels:

- **Agent prompts** explicitly instruct agents to never invent, infer, or assume clinical facts
- **Missing fields** are marked `[MISSING - FLAG FOR CLINICIAN REVIEW]`
- **Pending results** are marked `[PENDING - FLAG FOR CLINICIAN REVIEW]`
- **Conflicted fields** are marked `[CONFLICTED - FLAG FOR CLINICIAN REVIEW]`
- The output always ends with: *THIS IS A DRAFT. All flagged fields require clinician review.*
- The system never auto-finalizes a document

## Failure & Conflict Handling

- **PDF read failures** — reported per file, never silently skipped
- **Empty/unreadable pages** — marked `[EMPTY]` or `[ERROR]` in extraction
- **Conflicting values** — Conflict Detector identifies and escalates every disagreement between documents with source attribution
- **Undocumented medication changes** — flagged individually for reconciliation
- **Tool failures** — wrapped in try/except with fallback reporting
- **Step cap** — hard iteration limits prevent runaway agent loops

## Observability

Every agent run produces a structured trace file at `output/<patient_id>_trace.json` capturing:
- Step number and timestamp
- Agent reasoning
- Action taken
- Tool inputs
- Tool outputs
- Next decision

## Setup & Run

### Requirements
- Python 3.11+
- OpenAI API key (GPT-4o)

### Installation
```bash
git clone <repo>
cd discharge_agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
Create a `.env` file:
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o

### Add Patient Data
```
data/
└── patient_1/
├── admission_note.pdf
├── lab_results.pdf
└── ...
```

### Run
```bash
python main.py
```

Outputs saved to `output/`:
- `<patient_id>_discharge_summary.txt`
- `<patient_id>_trace.json`

## Cost Estimate
- PDF ingestion (vision): ~$0.40 per patient (71 pages)
- Agent LLM calls: ~$0.10 per patient
- **Total: ~$0.50 per patient**
- Cache hit on re-runs: ~$0.10 (vision skipped)

## Limitations & What I'd Do With More Time

### Current Limitations
- Patient name/DOB/ID often missing from scanned documents — these fields require manual entry
- Drug interaction checker uses a mock database — production would use OpenFDA or similar
- Escalation tool is mock — production would integrate with hospital EMR escalation workflows
- Handwritten text occasionally produces `[unclear]` tokens where GPT-4o cannot confidently read
- Single PDF per patient folder assumed — multi-file patients need folder-level organization


## Part 2 Results & Limitations

### Reward Signal
Edit distance between agent draft and reviewer-corrected version is used as the reward signal. Lower edit distance = less editing needed = better draft.

### Results (5 iterations, patient_2)
| Iteration | Edit Distance | Reward | Section Accuracy |
|-----------|--------------|--------|-----------------|
| 1 | 0.371 | 0.629 | 0.643 |
| 2 | 0.366 | 0.634 | 0.548 |
| 3 | 0.412 | 0.588 | 0.568 |
| 4 | 0.321 | 0.679 | 0.488 |
| 5 | 0.385 | 0.615 | 0.573 |

### Honest Assessment
The edit distance does not show a consistent downward trend across 5 iterations. This is expected given:
- The agent itself does not re-run between iterations — only the reviewer sees correction context
- GPT-4o is non-deterministic, causing natural variance
- 5 iterations with 1 patient is insufficient for meaningful learning
- True improvement requires the agent to re-generate drafts with injected correction memory

### What Would Work With More Time
- Re-run the full agent each iteration with correction context injected into the summary writer prompt
- Use 10+ patients and 10+ iterations for statistically meaningful results
- Fine-tune a smaller model on (draft, corrected) pairs using DPO/SFT
- Keep safety guardrails intact — optimize for structural improvement only, never factual invention

### Cold Start Problem
With only 1 patient and 5 iterations, the memory has too few examples to generalize. Meaningful learning requires at least 20-30 (draft, corrected) pairs across diverse patients.

### Gaming Risk
Optimizing to reduce edit distance can be gamed — an agent can lower edit distance by becoming vaguer or mimicking reviewer style rather than getting the medicine right. Safety guardrails (no fabrication, always flag missing data) must be preserved and evaluated independently of the reward signal.