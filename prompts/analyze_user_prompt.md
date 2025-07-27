# User prompt for AnalyzeNode - provides paper content with proper structure
user_prompt: |
  <context>
  Every distinct computational step or algorithm is a "node":
  - Main algorithms (Algorithm 1, Algorithm 2, etc.)
  - Sub-algorithms and subroutines
  - Individual steps within algorithms
  - Alternative strategies or methods
  - Any callable component
  </context>
  
  <task>
  Extract all algorithmic nodes from this paper and format according to the template.
  Show relationships between nodes (sequential, parallel, calls, choices).
  </task>
  
  <paper_content>
  {paper_content}
  </paper_content>