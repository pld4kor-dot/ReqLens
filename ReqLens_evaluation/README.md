# ReqLens Evaluation

**Evaluation pipeline for comparing Baseline, ReqInOne v1, and ReqLens v2 against poisoned benchmark datasets.**

Runs each system against Track 1 (hallucination detection) and Track 2 (noise detection) benchmark units, scores their outputs, and produces a structured comparison report.

---

## Systems Evaluated

| System ID | Description |
|-----------|-------------|
| `reqinone_v1` | ReqInOne-style-baseline v1 — LLM-based, no evidence gating |
| `reqlens_v2` | ReqLens v2 — multi-agent, evidence-gated pipeline |

---

## Prerequisites

**ReqLens_dataset_builder must be run first** to generate and poison benchmark units in `../ReqLens_dataset_builder/outputs/`.

---

## Setup

### 1. Create and activate the virtual environment

```bash
cd /path/to/ReqLens_evaluation

conda create -p ./eval_env python=3.11 -y
conda activate ./eval_env
```

### 2. Install packages

```bash
pip install -e .
pip install -e ../ReqLens       # required for the reqlens_v2 adapter
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in the required values:

```env
# Required — Azure OpenAI credentials
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_BASE_URL=https://<your-resource>.openai.azure.com/openai/v1/

# Model deployments
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5-mini
AZURE_OPENAI_JUDGE_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
REQINONE_V1_DEPLOYMENT=gpt-4.1-mini

# Paths — adjust if your folders are in a different location
BENCHMARK_OUTPUT_DIR=../ReqLens_dataset_builder/outputs
REQLENS_SRC_PATH=../ReqLens/src

# Output directory for evaluation results (default: outputs/)
EVAL_OUTPUT_DIR=outputs

LOG_LEVEL=INFO
```

---

## Running Evaluations

### Full run — all systems, both tracks, all units

```bash
reqlens-eval run
```

### Scoped by system

```bash
reqlens-eval run --system reqinone_v1
reqlens-eval run --system reqlens_v2
```

### Scoped by track

```bash
reqlens-eval run --track 1       # hallucination detection only
reqlens-eval run --track 2       # noise detection only
```

### Scoped by unit

```bash
reqlens-eval run --unit PROMISE_1
```

### Scoped by variant

```bash
reqlens-eval run --variant hallu_v1
```

### Combined filters

```bash
reqlens-eval run --system reqlens_v2 --track 1 --unit PROMISE_1
```

### Override benchmark directory at runtime

```bash
reqlens-eval run --benchmark-dir /custom/path/outputs
```

---

## Viewing Results

### List all saved evaluation runs

```bash
reqlens-eval list
```

### View a specific run report

```bash
reqlens-eval report --run-id <run-id>
```

Results are saved to `outputs/runs/<run-id>/` as structured JSON and a markdown report.

---

## Recommended Run Order

```bash
# 1. Smoke test to confirm setup
reqlens-eval run --system baseline --track 1 --unit PROMISE_1 --no-save

# 2. Full evaluation
reqlens-eval run

# 3. View results
reqlens-eval list
reqlens-eval report --run-id <run-id>
```
