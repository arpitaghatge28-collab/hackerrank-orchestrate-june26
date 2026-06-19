# Multi-Modal Evidence Review System

Verifies damage claims (car / laptop / package) using images, claim conversations, user history, and evidence requirements.

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
```

## Directory structure expected

```
code/
├── main.py
├── evaluation/
│   ├── evaluate.py
│   ├── evaluation_report.md
│   └── (eval_scores.json generated at runtime)
dataset/
├── claims.csv
├── sample_claims.csv
├── user_history.csv
├── evidence_requirements.csv
├── images/
│   ├── sample/
│   └── test/
```

> The `dataset/` folder must be a sibling of `code/` (or the working directory from which you run the scripts).

## Usage

### Step 1 — Evaluate on labeled sample data
```bash
cd code
python main.py sample         # runs pipeline on dataset/sample_claims.csv
python main.py eval           # scores output vs gold labels, prints accuracy
```

### Step 2 — Generate predictions for test set
```bash
python main.py                # runs on dataset/claims.csv → output.csv
# or explicitly:
python main.py test
```

## Output

`output.csv` is written to the working directory with these columns:

| Column | Description |
|--------|-------------|
| `user_id` | User identifier |
| `image_paths` | Semicolon-separated image paths (pass-through) |
| `user_claim` | Original conversation text (pass-through) |
| `claim_object` | car / laptop / package |
| `evidence_standard_met` | true/false — are images sufficient to evaluate? |
| `evidence_standard_met_reason` | Short reason |
| `risk_flags` | Semicolon-separated flags (or "none") |
| `issue_type` | Classified damage type |
| `object_part` | Affected part of the object |
| `claim_status` | supported / contradicted / not_enough_information |
| `claim_status_justification` | Image-grounded explanation |
| `supporting_image_ids` | Semicolon-separated image IDs (or "none") |
| `valid_image` | true/false — are images usable for automated review? |
| `severity` | none / low / medium / high / unknown |

## Configuration (top of main.py)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_WORKERS` | 3 | Parallel claim processing threads |
| `RETRY_LIMIT` | 3 | API call retries on failure |
| `RETRY_DELAY` | 5 | Base retry delay in seconds |
| `MODEL` | claude-sonnet-4-6 | Claude model to use |
| `MAX_TOKENS` | 1200 | Max output tokens per call |

## Log file

All processing is logged to `log.txt` (also printed to console). Submit this as the **chat transcript** requirement.

