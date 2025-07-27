"""Configuration constants for Paper2Code"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
PROMPTS_DIR = BASE_DIR / "prompts"

# Models
OPENAI_MODEL = "gpt-4.1-mini"
PERPLEXITY_MODEL = "sonar"

# API Settings
PERPLEXITY_TIMEOUT = 120
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"