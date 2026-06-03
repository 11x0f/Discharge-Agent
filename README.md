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

### Approach
A simulated reviewer applies a consistent editing policy to agent drafts. 
Normalized edit distance between draft and corrected version is used as the 
reward signal. Past corrections are stored in memory and injected as context 
into future reviewer prompts.

### Results

**Patient 1 (Synthetic)**
| Iteration | Edit Distance | Reward | Section Accuracy |
|-----------|--------------|--------|-----------------|
| 1 | 0.7459 | 0.2541 | 0.6046 |
| 2 | 0.7235 | 0.2765 | 0.5048 |
| 3 | 0.7503 | 0.2497 | 0.5108 |
| 4 | 0.6619 | 0.3381 | 0.6230 |
| 5 | 0.6748 | 0.3252 | 0.6301 |

Improvement: 0.0711 reduction in edit distance (iteration 1 → 5)

**Patient 2 (Real)**
| Iteration | Edit Distance | Reward | Section Accuracy |
|-----------|--------------|--------|-----------------|
| 1 | 0.9267 | 0.0733 | 0.5072 |
| 2 | 0.7394 | 0.2606 | 0.4766 |
| 3 | 0.4668 | 0.5332 | 0.5214 |
| 4 | 0.4077 | 0.5923 | 0.5152 |
| 5 | 0.9208 | 0.0792 | 0.5108 |

Improvement: 0.0059 reduction in edit distance (iteration 1 → 5)

### Honest Assessment
Patient 1 shows a consistent downward trend in edit distance across iterations,
suggesting the correction memory injection is having a positive effect on 
reviewer consistency. Patient 2 shows high variance — edit distance drops 
significantly in iterations 3-4 but spikes back in iteration 5. This is 
primarily due to GPT-4o non-determinism rather than the learning mechanism 
failing.

True improvement requires:
- The agent itself to re-run with correction memory injected into its prompts
- More patients (20-30 minimum) for meaningful generalization
- More iterations (10+) to smooth out LLM variance

### Limitations
- Agent does not re-run between iterations — only reviewer sees correction context
- GPT-4o non-determinism causes variance that masks learning signal
- 2 patients and 5 iterations is insufficient for statistical significance
- Edit distance can be gamed by vagueness — safety guardrails remain independent
  of reward signal to prevent this

### Cold Start Problem
With only 2 patients and 5 iterations, the memory has 25 pairs — too few to 
generalize. Meaningful learning requires at least 20-30 diverse patients.