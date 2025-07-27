# CLAUDE.md

Paper2Code extracts algorithms from STEM papers using a 2-node async pipeline.

## Architecture

1. **BuildOverview** (OpenAI gpt-4.1-mini): Single API call generates complete algorithm with Mermaid diagram
2. **ProcessNode** (AsyncParallelBatchNode): Parallel research for each node using Perplexity

## Commands

```bash
source .venv/bin/activate

# Single paper
python src/paper2code.py input/markdown/test_samples/<paper_name>/main.md

# Batch run
python utils/run_all_papers.py
```

## Key Details

- Node format: `1. [[Node Name]]` (numbered list with wiki-links)
- Templates/prompts: YAML frontmatter files loaded via `load_yaml_field()`
- LaTeX: Use `$` notation only
- Parallel execution: Uses `call_openai_async()` in ProcessNode

## Requirements

- `OPENAI_API_KEY` and `PERPLEXITY_API_KEY` in `.env`
- Python 3.8+, managed by `uv`