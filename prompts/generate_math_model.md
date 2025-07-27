# Prompt for GenerateMathModel - extracts math and algorithm fundamentals
prompt: |
  Read the paper strategically in this order: Abstract → Figures → Introduction → Discussion → Methods/Results
  
  Focus on understanding the overall algorithm and its mathematical foundation.
  
  Paper content:
  {paper_content}
  
  Generate the mathematical overview following this exact template:
  {math_model_template}
  
  CRITICAL RULES:
  - Use double curly braces {{}} for placeholders you fill in
  - ALL math symbols MUST use $ notation: $\pi$, $\theta$, $x_t$
  - NEVER use \( \) or backticks for math
  - Block equations: Use $$ on separate lines
  - Focus on the PRIMARY mathematical model/equation
  
  Also return a JSON list of all major algorithmic nodes/components you identify:
  {{"nodes": ["Node Name 1", "Node Name 2", ...]}}