# Structured Text Classification & Information Extraction with a Local LLM

NLP course, Project I. IMDB reviews are classified (sentiment) and person names are extracted by a local LLM (Ollama, `llama3.2:3b`, Q4_K_M). The output must be valid JSON. Two settings are compared: **prompt-only** and **schema-constrained** (Ollama `format` with a JSON schema), each at temperature 0 and 1.0. No cloud APIs are used.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
```

Ollama 0.40.2, Python `ollama` package 0.6.3 (see `env_info.txt`). Generation parameters: `num_ctx=4096`, `seed=42`, temperature 0 and 1.0.

## Run

| Step | Command | Output |
|---|---|---|
| Main experiment (100 reviews x 2 settings x 2 temperatures, about 10 min) | `python model_v2.py` | `results_*.csv`, `summary_llama3.2_3b.csv` |
| Sentiment and validity analysis | `python analyze_v2.py` | console |
| Entity F1 against gold labels | `python entity_f1.py` | console |
| Cleaning function test (synthetic broken outputs) | `python test_clean.py` | console |
| Live demo (same samples in both settings, new samples) | `python live_demo.py` | console |


## Files

| File | Purpose |
|---|---|
| `model_v2.py` | Prompt, schema, Ollama calls, cleaning/parsing, validity checks |
| `analyze_v2.py` | Macro F1, confusion matrices, validity errors, setting comparison |
| `annotate.txt` | Manually annotated gold person names (100 reviews, first 1500 characters) |
| `entity_f1.py` | Entity precision/recall/F1 (exact and partial match) |
| `test_clean.py` | Tests `clean_output` on synthetic broken model outputs |
| `range_probe.py` | Probe: does the schema enforce the score range? |
| `check_mixed.py` | Shows the cases where the schema forced `mixed` to `negative` |
| `live_demo.py` | Live test: dataset samples in both settings and two temperatures, plus new samples |
| `results_*.csv` | Raw model outputs and per-review checks for each setting and temperature |
| `env_info.txt` | Ollama and model details (version, quantization) |

## Data

`stanfordnlp/imdb`, test split, shuffled with seed 42, 50 positive + 50 negative reviews. Each review is truncated to its first 1500 characters before it is sent to the model.