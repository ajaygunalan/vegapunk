# Prompt for IdentifyNode - finds computational nodes in papers
prompt: |
  You are a technical specialist who identifies algorithmic nodes in STEM papers.
  
  A "node" is a distinct concept that captures a unit of meaning, worthy of its own focused explanation (up to 500 words). Extract the MOST IMPORTANT computational nodes from the paper.
  
  Paper content:
  {paper_content}
  
  Return results as JSON:
  {{
    "nodes": [
      {{
        "name": "Exact Algorithm Name from Paper",
        "description": "What this node does in 1-2 sentences"
      }}
    ],
    "summary": "One paragraph overview of the paper's main contribution"
  }}
  
  Rules:
  - Extract ONLY the 5-8 most important algorithms, methods, or core computational components
  - Focus on major contributions, not minor helper functions or trivial steps
  - Each node should represent a substantial concept worthy of detailed explanation
  - Use exact names from the paper when available
  - Node names MUST only contain letters, numbers, and spaces - NO slashes, colons, or special characters
  - Example good names: "Collision Affording Point Tree", "Nearest Neighbor Search", "Construction Algorithm"
  - Example bad names: "Helper Function", "Step 1", "Initialize Variables"