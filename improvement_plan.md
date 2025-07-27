# Paper2Code Improvement Plan

## Executive Summary

**Problem**: Monolithic architecture with one massive API call containing full paper + templates.

**Solution**: Modular 5-node pipeline with focused API calls.

**Impact**: Lower costs, better outputs, easier debugging.

## Architecture Design

### Current Issues
1. **AnalyzeNode does everything** - identifies nodes, formats, generates queries in ONE call
2. **No flow framework** - hardcoded pipeline in main()
3. **Expensive** - sends entire paper + templates every time

### New Architecture: 4 Specialized Nodes

```
Paper → [Identify] → [Analyze] → [Query] → [Research] → Output
         GPT-4.1      GPT-4.1     GPT-4.1    Perplexity
```

1. **Identify** - Find all computational nodes from paper
2. **Analyze** - Map relationships, create & save algorithm_overview.md
3. **Query** - Generate research questions for each node
4. **Research** - Parallel research via Perplexity (BatchNode) + save results

### Core Components

```python
# Flow framework (from PocketFlow)
class Node:
    def prep(self, shared): pass
    def exec(self, prep_res): pass  
    def post(self, shared, prep_res, exec_res): pass

# Shared state
shared = {
    # Config
    "max_nodes": 15,
    "model": "gpt-4.1",
    
    # State
    "paper_content": str,
    "nodes": list,
    "relationships": dict,
    "queries": dict,
    "research_results": dict
}

# Node chaining
identify >> analyze >> query >> research
```

## Implementation Plan

### High Priority (Week 1-2)
1. Create base Node/Flow classes
2. Split AnalyzeNode into 4 nodes
3. Implement shared state management
4. Add BatchNode for parallel research

### Medium Priority (Week 3-4)
1. Optimize prompts per node
2. Add retry/error handling
3. Implement caching

## Why This Architecture

**From PocketFlow**: Node pattern, flow engine, shared state, batch processing

**Our Focus**: Algorithm extraction, specialized prompts, Perplexity research

**Result**: Modular, testable, efficient pipeline optimized for papers.

## Prompt Refactoring Plan

### Current State
- **analyze_system_prompt.md**: Does everything (identify + overview + queries)
- **analyze_user_prompt.md**: Single prompt for all tasks
- **research_system/user_prompt.md**: Keep as-is (already focused)

### New Prompt Structure

1. **identify_system/user_prompt.md** (NEW)
   - Focus: Find all computational nodes in paper
   - Input: Full paper markdown
   - Output: JSON with nodes list and descriptions
   - Extract from current analyze prompt: Lines 22-27 (node definition)

2. **analyze_system/user_prompt.md** (REFACTOR)
   - Focus: Create algorithm overview with relationships
   - Input: Node list + full paper
   - Output: Formatted algorithm_overview.md using template
   - Keep from current: Template formatting, LaTeX rules, data flow mapping

3. **query_system/user_prompt.md** (NEW)
   - Focus: Generate targeted research questions
   - Input: Each node + paper context
   - Output: 3-5 specific questions per node
   - Extract from current: Query generation logic

4. **research_system/user_prompt.md** (NO CHANGE)
   - Already well-focused on researching individual nodes

### Template Changes
- **algorithm_overview.md**: Keep as-is (used by Analyze node)
- **algorithm_step.md**: Keep as-is (used by Research node)
- **node_query.md**: Keep as-is (used by Query node for question structure)

### Key Insight
The current analyze prompt is doing 3 jobs. By splitting it, each node can:
- Use smaller, focused prompts
- Be tested independently
- Produce better quality outputs