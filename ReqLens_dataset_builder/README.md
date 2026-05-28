# ReqLens Dataset Builder

**Benchmark dataset generator for the ReqLens evaluation pipeline.**

Builds structured, evidence-grounded benchmark units from the PROMISE and PURE datasets, then poisons them to create adversarial test cases for evaluating requirements engineering systems.

---

## Output Structure

```
outputs/
├── promise/
│   └── PROMISE_<n>/
│       └── unit.json          
├── pure/
│   └── PURE_<n>/
│       └── unit.json
└── poisoned/
    ├── track1/
    │   └── PROMISE_<n>/
    │       └── poisoned_track1_hallu_v<n>.json   # hallucinated requirements injected
    └── track2/
        └── PROMISE_<n>/
            └── poisoned_track2_v<n>.json         # contradictions + duplicates injected
```

---

## Setup

### 1. Create and activate the virtual environment

```bash
cd /path/to/ReqLens_dataset_builder

conda create -p ./env python=3.11 -y
conda activate ./env
```

### 2. Install the package

```bash
pip install -e .
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

# Model deployments — set to your actual deployment names
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5
AZURE_OPENAI_REASONING_DEPLOYMENT=gpt-5-mini
AZURE_OPENAI_EXTRACTION_DEPLOYMENT=gpt-5-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Output directory (default: outputs/)
OUTPUT_DIR=outputs

# PROMISE dataset settings
PROMISE_INPUT=data/promise/promise.csv
PROMISE_HAS_HEADER=true
PROMISE_PROJECT_COL=ProjectID
PROMISE_TEXT_COL=RequirementText
PROMISE_LABEL_COL=class
PROMISE_MAX_PROJECTS=15
PROMISE_MAX_REQS_PER_PROJECT=100

# PURE dataset settings
PURE_INPUT_DIR=data/pure
PURE_MAX_DOCS=10
```

### 4. Place raw datasets

```
data/
├── promise/
│   └── promise.csv       ← PROMISE dataset CSV
└── pure/
    └── *.txt / *.pdf     ← PURE document files
```

---

## Generating Datasets

### Generate from PROMISE only

```bash
reqlens-benchmark-builder promise
```

### Generate from PURE only

```bash
reqlens-benchmark-builder pure
```

### Generate from both datasets

```bash
reqlens-benchmark-builder both
```

### Override output directory at runtime

```bash
reqlens-benchmark-builder both --output-dir /custom/path/outputs
```

### Set log level

```bash
reqlens-benchmark-builder both --log-level DEBUG
```

---

## Poisoning

Poisoning reads the generated `outputs/` directory and injects adversarial artifacts. **Run dataset generation first.**

### Poison all units, both tracks

```bash
reqlens-benchmark-builder poison
# equivalent to: --track both
```

### Poison track 1 only (hallucinated requirements)

```bash
reqlens-benchmark-builder poison --track 1
```

### Poison track 2 only (contradictions + duplicates)

```bash
reqlens-benchmark-builder poison --track 2
```

### Poison a specific unit only

```bash
reqlens-benchmark-builder poison --unit PROMISE_1
```

### Custom poison parameters

```bash
# Track 1: inject 5 hallucinated requirements per unit
reqlens-benchmark-builder poison --track 1 --hallucinations 5

# Track 2: inject 2 contradictions and 2 duplicates per unit
reqlens-benchmark-builder poison --track 2 --contradictions 2 --duplicates 2

# Combine: specific unit, specific track, custom counts
reqlens-benchmark-builder poison --track 1 --unit PROMISE_1 --hallucinations 5
```

---

## Tracks Explained

| Track | Type | What is injected | Evaluates |
|-------|------|-----------------|-----------|
| Track 1 | Hallucination | Fake requirements with no source evidence | Whether the system rejects unsupported requirements |
| Track 2 | Noise | Contradictory and duplicate requirements | Whether the system detects inconsistencies |

---

## Full Run (recommended order)

```bash
reqlens-benchmark-builder both                                        # generate silver units
reqlens-benchmark-builder poison --track both                         # poison all units
```
