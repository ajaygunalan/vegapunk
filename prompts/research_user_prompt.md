# User prompt template for ResearchNode - combines node query with step template
user_prompt: |
  <context>
  You're teaching someone who wants to understand, not just implement.
  </context>
  
  <requirements>
  - Explain WHY this step is needed and WHAT problem it solves
  - Use intuitive explanations before mathematical formulas
  - Define any technical terms in simple language
  - Show the reasoning behind design choices
  - Fill EVERY section of the template thoughtfully
  - Replace ALL {{placeholders}} with actual content
  </requirements>
  
  <task>
  Help a student understand this algorithm step by explaining its purpose and how it works.
  Fill out the template to create a clear learning resource.
  </task>
  
  <algorithm_query>
  {query}
  </algorithm_query>
  
  <output_template>
  {step_template}
  </output_template>