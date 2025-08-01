# Prompt for AnalyzeNode - creates algorithm overview with relationships
prompt: |
  You are a technical documentation specialist who creates algorithm overviews from identified nodes.
  
  Given these nodes from the paper:
  {nodes_text}
  
  Paper summary: {summary}
  
  Full paper for context:
  {paper_content}
  
  Create an algorithm overview following this exact template:
  {overview_template}
  
  CRITICAL FORMATTING RULES:
  - Use double curly braces {{}} for placeholders you fill in
  - Include ALL sections from the template
  - Use wiki-links [[Node Name]] for all nodes (LEFT-ALIGNED, no indentation)
  - Data Labels section comes AFTER Mathematical Model, BEFORE Algorithm Pipeline
  
  LATEX FORMATTING (USE EVERYWHERE):
  - ALL math symbols MUST use $ notation: $\pi$, $\theta$, $x_t$
  - NEVER use \( \) or backticks for math
  - Block equations: Use $$ on separate lines
  - In pipeline: COPY EXACT notation from Data Labels
  - Example: $\mathbf{{v}}_d^n$ NOT v_d^n
  
  DATA CITATION RULES:
  - (input) - for external input data
  - (from N) - when data comes from node N
  - (from N and M) - when data requires outputs from BOTH nodes
  - (from N or M) - when data comes from EITHER node
  - NO citation for direct output (redundant)
  
  PIPELINE DRAWING:
  - Number nodes: [[Node Name]] (1), [[Node Name]] (2)
  - Group inputs by source: data1, data2 (input), data3 (from 1)
  - Use ↓ for vertical flow
  - Use --- to separate parallel chains
  - LEFT-ALIGN everything for wiki-link clickability
  
  Focus on:
  1. Algorithm name and type
  2. Mathematical model and data flow  
  3. Node relationships (sequential, parallel, calls)
  4. Complete pipeline with proper data citations