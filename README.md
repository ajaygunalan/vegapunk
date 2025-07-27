# Paper2Code

Extract algorithms from papers → structured knowledge using AI.

## Setup
```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Install marker for PDF extraction (optional)
uv pip install marker-pdf

# Copy .env.example to .env and add your API keys
cp .env.example .env
# Required: OPENAI_API_KEY and PERPLEXITY_API_KEY
```

## Usage

### Single Paper Processing
```bash
python src/paper2code.py <paper_markdown_file>

# Example
python src/paper2code.py input/markdown/test_samples/your_vit_is_secretly_a_hybrid_discriminativegenera/main.md

# Output automatically goes to: output/your_vit_is_secretly_a_hybrid_discriminativegenera/
```

### Batch Processing
```bash
# Run all test samples with timing
python utils/run_all_papers.py
```

### PDF Extraction
```bash
# Extract single PDF
bash utils/pdf_extract.sh path/to/paper.pdf input/markdown/

# Extract all PDFs in a directory
for pdf in pdf/*.pdf; do
    bash utils/pdf_extract.sh "$pdf" input/markdown/
done
```

## Directory Structure
```
├── src/paper2code.py     # Main algorithm extraction
├── templates/            # YAML templates for prompts
├── prompts/             # System and user prompts
├── input/
│   ├── pdf/             # Source PDFs
│   └── markdown/        # Extracted markdown
│       └── test_samples/# Selected test papers
├── output/              # Algorithm outputs
│   └── <paper_name>/    # Per-paper results
└── utils/              
    ├── pdf_extract.sh    # PDF to markdown converter
    └── run_all_papers.py # Batch processor with timing
```

## Output Structure
```
output/
└── <paper_name>/
    ├── algorithm_overview.md
    ├── analyze_raw.md
    └── <algorithm_step>.md
```