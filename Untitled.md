# Paper2Code Improvement Plan

## Executive Summary

**Core Problem**: Our current architecture sends the ENTIRE paper + templates in ONE massive API call, making it expensive, inflexible, and hard to debug.

**Solution**: Adopt PocketFlow's modular architecture with 5 specialized nodes, each doing ONE focused task with smaller API calls.

**Key Benefits**:
- Reduced token usage (smaller, focused API calls)
- Better quality outputs (focused prompts)
- Easy debugging (test individual nodes)
- Lower costs per paper

## Current Architecture Issues

### 1. **Monolithic API Call Problem**
- **Issue**: AnalyzeNode sends ENTIRE paper + both templates in ONE massive call
- **Impact**: 
  - Huge token usage (costly)
  - Single point of failure
  - Hard to debug/iterate
  - No granular control

### 2. **Lack of Flow Architecture**
- **Issue**: Hardcoded pipeline in main(), no reusable flow framework
- **PocketFlow Solution**: Clean Flow engine with node chaining (`>>`)
- **Our Gap**: Just two classes called sequentially

### 3. **Poor Separation of Concerns**
- **AnalyzeNode does too much**:
  - Identifies nodes
  - Formats overview
  - Generates queries
  - All in one API call!

### 4. **Model Selection**
- Currently using "o4-mini" 
- **Decision: Standardize on GPT-4.1 for ALL OpenAI calls**
  - Use GPT-4.1 for nodes 1-4 (identify, analyze, query)
  - Keep Perplexity API for ResearchNodes (parallel research)
  - Consistent high quality for algorithm understanding
  - Worth the cost for better accuracy

## Proposed New Architecture

### Phase 1: Implement Flow Framework
```python
# Minimal flow engine
class Node:
    def prep(self, shared): pass
    def exec(self, prep_res): pass  
    def post(self, shared, prep_res, exec_res): pass

class Flow:
    def __init__(self, start):
        self.start = start
    
    def run(self, shared):
        # Execute nodes in sequence
```

### Phase 2: Break Down into 5 Focused Nodes

1. **Identify**  
   - Input: Full paper markdown
   - Task: Identify all computational nodes (algorithms, steps, methods)
   - Output: Node list with descriptions + overview

2. **Analyze**
   - Input: Node list + full paper markdown
   - Task: Map data flow and dependencies between nodes
   - Output: Relationships, data flow, and algorithm_overview.md

3. **Query**
   - Input: Nodes + relationships + full paper markdown
   - Task: Generate research questions for each node
   - Output: Node-query mapping

4. **Research** (BatchNode)
   - Input: Node queries
   - Task: Research each node in parallel via Perplexity
   - Output: Detailed explanations

5. **Combine**
   - Input: All research results from Research node
   - Task: Save each node's research to individual .md files
   - Output: Node files written to output directory (no LLM)

### Phase 3: Optimize API Calls

**Current (BAD)**:
```
1 massive call: Full paper + Both templates + System prompt = Very large token usage
```

**Proposed (GOOD)**:
```
Node 1: Identify - Full paper (largest call)
Node 2: Analyze - Smaller focused prompt  
Node 3: Query - Small prompt
Node 4: Research - Small prompts (parallel)
Node 5: Combine - No LLM needed
Result: Multiple smaller calls instead of one huge call
```

### Phase 4: Configuration & State Management

```python
shared = {
    # Configuration (set once at start)
    "max_nodes": 15,
    "model": "gpt-4.1",
    "include_patterns": ["Algorithm", "Procedure", "Method"],
    "use_cache": True,  # For medium priority caching
    "max_retries": 5,   # For medium priority retry handling
    
    # State (updated by nodes)
    "paper_content": str,
    "nodes": list,
    "relationships": dict,
    "queries": dict,
    "research_results": dict
}
```

## Benefits of New Architecture

1. **Modularity**: Each node has single responsibility
2. **Debuggability**: Can test/fix individual nodes
3. **Efficiency**: Smaller, focused API calls
4. **Flexibility**: Easy to swap models per node
5. **Extensibility**: Add new nodes without touching others
6. **Cost Savings**: Lower token usage from focused prompts
7. **Better Quality**: Focused prompts = better outputs

## Implementation Priority

1. **High Priority** (Do First):
   - Implement basic Flow framework
   - Split AnalyzeNode into 5 focused nodes
   - Add shared state management with configuration (Phase 4)
   - Implement BatchNode for parallel research

2. **Medium Priority** (Do Later):
   - Optimize prompts for each node
   - Add retry/error handling (max_retries, wait time)
   - Add caching with use_cache flag

## Model Configuration

- **Identify**: GPT-4.1
- **Analyze**: GPT-4.1
- **Query**: GPT-4.1
- **Research**: Perplexity Sonar API (parallel calls)
- **Combine**: No LLM needed (just file I/O)

## Concrete Implementation Example

```python
# flow.py - New modular architecture
from base import Node, Flow, BatchNode

class IdentifyNode(Node):
    """Identify computational nodes from full paper"""
    
    def prep(self, shared):
        return {
            "paper_content": shared["paper_content"],
            "max_nodes": shared.get("max_nodes", 15)
        }
    
    def exec(self, prep_res):
        prompt = f"""
        Identify all computational nodes in this paper.
        A node is any algorithm, step, method, or procedure.
        Maximum {prep_res["max_nodes"]} nodes.
        
        Paper:
        {prep_res["paper_content"]}
        
        Return nodes list and overview...
        """
        # API call to GPT-4.1
        return result
    
    def post(self, shared, prep_res, exec_res):
        shared["nodes"] = exec_res["nodes"]
        shared["overview"] = exec_res["overview"]

# Usage with clean names
def create_paper2code_flow():
    identify = IdentifyNode()
    analyze = AnalyzeNode()
    query = QueryNode()
    research = ResearchNode()  # BatchNode
    combine = CombineNode()
    
    # Chain nodes
    identify >> analyze >> query >> research >> combine
    
    return Flow(start=identify)
```

## Migration Strategy

1. **Week 1**: Implement base Node/Flow classes
2. **Week 2**: Split AnalyzeNode into 5 focused nodes
3. **Week 3**: Add remaining nodes and optimize
4. **Week 4**: Testing and refinement

## Expected Improvements

- **Token Usage**: Reduced through focused prompts
- **Modularity**: 2 classes → 5 focused nodes
- **Debuggability**: Can test each node independently
- **Flexibility**: Easy to swap models/prompts per node
- **Cost**: Lower per paper through efficiency

## Key Decision: Hybrid Approach

We adopt PocketFlow's **patterns** while keeping our **domain focus**:

### What We Take from PocketFlow:
- Node abstraction (prep, exec, post methods)
- Flow engine with `>>` chaining
- Shared state dictionary
- BatchNode for parallel processing
- Retry/caching mechanisms

### What We Keep Unique:
- Algorithm extraction focus (not tutorials)
- Our specialized prompts/templates
- Node structure optimized for papers
- Perplexity for research

### Our 5 Nodes:
1. **Identify** - Find nodes from full paper
2. **Analyze** - Map relationships & data flow
3. **Query** - Generate research questions
4. **Research** - Parallel Perplexity calls
5. **Combine** - Assemble final output

This gives us proven architecture with our specialized purpose.