# Prompt for GenerateNodeDetails - writes detailed node descriptions
prompt: |
  Given the algorithm overview and pipeline, write detailed descriptions for each node.
  
  Algorithm: {algorithm_name}
  Nodes: {nodes_list}
  
  Math overview:
  {math_overview}
  
  Pipeline:
  {pipeline_overview}
  
  Paper content for deep understanding:
  {paper_content}
  
  Generate node details following this exact template:
  {node_details_template}
  
  For EACH node, provide:
  - One line summary of what it does
  - Challenge: What makes this step hard/non-trivial
  - Solution: The specific approach used
  - Key Insight: The mathematical/algorithmic realization
  
  Use wiki-links [[Node Name]] for all node names.
  
  Return the completed template.