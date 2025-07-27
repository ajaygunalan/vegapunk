# Prompt for GeneratePipeline - creates data flow and pipeline
prompt: |
  Given the algorithm name and nodes, create the data labels and pipeline sections.
  
  Algorithm: {algorithm_name}
  Nodes identified: {nodes_list}
  
  Paper content for context:
  {paper_content}
  
  Generate the pipeline following this exact template:
  {pipeline_template}
  
  CRITICAL RULES:
  - Identify all data (input/intermediate/output) used by the nodes
  - Create pipeline showing how nodes connect through data
  - Use wiki-links [[Node Name]] for all nodes
  - LEFT-ALIGN everything (no indentation)
  - Data names MUST use exact LaTeX notation
  - Follow all citation rules in template
  
  Return the completed template sections.