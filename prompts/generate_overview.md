# Prompt for generating complete algorithm overview with Mermaid diagram
prompt: |
  Analyze this academic paper and extract the complete algorithm structure.
  
  Paper content:
  {paper_content}
  
  Generate a comprehensive algorithm overview following this EXACT template:
  {template}
  
  CRITICAL INSTRUCTIONS:
  1. Start with algorithm name as ## heading
  2. Write compelling overview paragraph covering problem, challenge, and key insight
  3. Mathematical Model section with primary equation and symbol definitions
  4. Data Labels section organizing Input/Intermediate/Output data
  5. Algorithm Pipeline as a Mermaid diagram showing data flow through nodes
  6. End with numbered Nodes list using [[Node Name]] format
  
  MERMAID DIAGRAM RULES:
  - Use quotes for all labels to avoid parsing issues
  - Show data flow with arrows and edge labels
  - Include input data, nodes, and output data
  - Style with classDef for visual clarity
  
  NODE EXTRACTION:
  - List nodes at the end as: 1. [[Node Name]] - one line summary
  - Include 5-8 key algorithmic components
  - Use descriptive names that explain the node's purpose
  
  FORMATTING:
  - Use $ for all math notation
  - Keep descriptions crisp and technical
  - Ensure Mermaid syntax is valid
  - Follow the template structure exactly