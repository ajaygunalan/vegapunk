# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Paper2Code is a two-stage AI pipeline that extracts algorithmic structures from STEM papers and generates educational documentation for each algorithm component.

## Architecture

The system follows a two-node architecture:

1. **AnalyzeNode** (OpenAI o4-mini): Extracts algorithm structure from papers
   - Parses paper markdown → identifies all algorithmic "nodes" (algorithms, steps, subroutines)
   - Generates overview following algorithm_overview.md template
   - Creates node queries for each identified component

2. **ResearchNode** (Perplexity Sonar): Researches and documents each node
   - Takes node queries → generates educational explanations
   - Parallel processing for all nodes
   - Outputs follow algorithm_step.md template

## Commands

```bash
# Activate virtual environment (required)
source .venv/bin/activate

# Process single paper
python src/paper2code.py input/markdown/test_samples/<paper_name>/main.md

# Run all test samples with timing
python utils/run_all_papers.py

# Extract PDF to markdown (requires marker-pdf)
bash utils/pdf_extract.sh path/to/paper.pdf input/markdown/
```

## Critical Implementation Details

1. **Prompt/Template Loading**: All prompts and templates are Markdown files with YAML frontmatter. The `load_yaml_field()` function extracts the specific field from these files.

2. **Node Extraction**: The NODES section in analyze_raw.md uses a specific format:
   ```
   ===Node Name===
   Query: |
     Questions about the node
   ```

3. **Wiki-link Format**: All node references use `[[Node Name]]` format (Obsidian-compatible). These must be left-aligned for clickability.

4. **LaTeX in Markdown**: Use `$` notation everywhere (never `\(` or backticks). Copy exact notation from Data Labels to Pipeline sections.

5. **Data Citations**: Every data item in pipelines must cite its source using formats like `(from 1)`, `(from 1 or 2)`, `(from 1 and 2)`.

## File Structure Conventions

- Prompts use XML tags for structure: `<role>`, `<instructions>`, `<constraints>`, `<examples>`, `<templates>`
- Templates define exact output format - never deviate from template structure
- Output files are named after node names (spaces allowed in filenames)

## Environment Requirements

- Python 3.8+
- API Keys: `OPENAI_API_KEY` and `PERPLEXITY_API_KEY` in `.env`
- Virtual environment managed by `uv`