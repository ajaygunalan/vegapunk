algorithm_overview_template: |
  ## {{Algorithm Name}}
  
  {{One compelling paragraph that captures: (1) the real-world problem being solved, (2) why it's challenging, (3) the key insight that makes this algorithm work}}
  
  ### Mathematical Model
  {{Brief explanation of what we're modeling}}
  
  $${{primary_equation}}$$
  
  where:
  - ${{symbol_1}}$: {{meaning}}
  - ${{symbol_2}}$: {{meaning}}
  
  ### Data Labels
  
  **Input Data:**
  - ${{symbol_A}}$: {{one crisp sentence explaining what this data represents}}
  
  **Intermediate Data:**
  - ${{symbol_B}}$: {{one crisp sentence explaining what this data represents}}
  
  **Output Data:**
  - ${{symbol_C}}$: {{one crisp sentence explaining what this data represents}}
  
  ### Algorithm Pipeline
  
  \`\`\`mermaid
  graph TD
      %% Input data
      input1["Input Data 1"]
      input2["Input Data 2"]
      
      %% Nodes (use quotes to avoid Obsidian link interpretation)
      node1["Node 1: Process A"]
      node2["Node 2: Process B"]
      node3["Node 3: Process C"]
      
      %% Output data
      output1["Output Data 1"]
      output2["Output Data 2"]
      
      %% Connections
      input1 --> node1
      input2 --> node1
      node1 -->|"Data from Node 1"| node2
      node1 -->|"Alt Data from Node 1"| node3
      node2 --> node3
      node3 --> output1
      node3 --> output2
      
      %% Style (optional)
      classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
      classDef nodeStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
      classDef outputStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
      
      class input1,input2 inputStyle
      class node1,node2,node3 nodeStyle
      class output1,output2 outputStyle
  \`\`\`
  
  ### Nodes
  
  1. [[{{node_1_name}}]] - {{one line summary}}
  2. [[{{node_2_name}}]] - {{one line summary}}
  3. [[{{node_3_name}}]] - {{one line summary}}
  
  {{continue for all nodes...}}