# Prompt for QueryNode - generates research questions for nodes
prompt: |
  You generate focused research questions about algorithms and computational methods.
  
  For the algorithm/method "{node_name}" which {node_description}, generate 3-5 specific research questions.
  
  Questions should help someone understand:
  - What it does conceptually
  - How it works internally  
  - When to use it
  - Common variations or improvements
  - Practical implementation details
  
  Format your response following this template:
  {query_template}
  
  Make questions specific and technical, suitable for researching via academic sources.