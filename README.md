# Paper2Code

AI pipeline that extracts algorithms from STEM papers in ~30s.

## Setup
```bash
# Clone & setup
git clone https://github.com/ajaygunalan/vegapunk.git
cd vegapunk
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Configure
cp .env.example .env
# Add OPENAI_API_KEY and PERPLEXITY_API_KEY
```

## Usage

```bash
# Process paper
python src/paper2code.py input/markdown/test_samples/your_vit_is_secretly_a_hybrid_discriminativegenera/main.md

# Batch run
python utils/run_all_papers.py

# PDF → Markdown (requires: uv pip install marker-pdf)
bash utils/pdf_extract.sh paper.pdf input/markdown/
```

## Output

Each paper generates:
- `algorithm_overview.md` - Full algorithm with Mermaid diagram
- `<node_name>.md` - Educational explanation for each algorithm component

## Architecture

2-node async pipeline using PocketFlow:
1. **BuildOverview** - Extracts algorithm structure (OpenAI)
2. **ProcessNode** - Parallel research for each node (Perplexity)