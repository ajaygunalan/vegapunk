# Code Generation Architecture

## Current State

We have a 2-node async pipeline:
- **BuildOverview**: Generates algorithm overview with nodes list
- **ProcessNode**: Parallel research for each node

## Proposed Architecture

### High-Level Design

```
Paper → BuildOverview → PlanGeneration → AnalyzeFiles → CodeGeneration
           ↓                  ↓              ↓              ↓
      Nodes + Diagram    Architecture    Logic specs    Complete repo
                          File list      (parallel)      (parallel)
                          Dependencies
```

### Key Improvements

1. **PlanGeneration** (after BuildOverview):
   - Takes nodes list from BuildOverview
   - Generates file architecture
   - Identifies dependencies between files
   - Outputs structured plan

2. **AnalyzeFiles** (AsyncParallelBatchNode):
   - Parallel logic analysis for each file
   - Uses context7 to fetch relevant library docs
   - Generates detailed specs per file

3. **CodeGeneration** (AsyncParallelBatchNode):
   - Parallel code generation
   - Uses specs from AnalyzeFiles
   - Respects dependency order in shared state

## Implementation Plan

### 1. PlanGeneration Node

```python
class PlanGeneration(Node):
    """Generate file architecture from algorithm nodes"""
    
    def prep(self, shared):
        print("\n📋 Plan Generation: started ...")
        shared["plan_start"] = time.time()
        return {
            "nodes": shared["nodes"],
            "algorithm_overview": shared["algorithm_overview"],
            "paper_content": shared["paper_content"]
        }
    
    def exec(self, context):
        prompt = load_yaml_field('prompts/plan_generation.md', 'prompt').format(
            nodes=context["nodes"],
            overview=context["algorithm_overview"],
            paper=context["paper_content"]
        )
        
        response = call_llm(prompt, model="claude-3-5-sonnet")
        
        # Parse response into structured format
        plan = {
            "files": [...],  # List of files to generate
            "dependencies": {...},  # File dependency graph
            "file_purposes": {...},  # What each file does
            "shared_interfaces": {...}  # Common data structures
        }
        
        return plan
```

### 2. AnalyzeFiles Node

```python
class AnalyzeFiles(AsyncParallelBatchNode):
    """Analyze each file in parallel with context7 docs"""
    
    async def prep_async(self, shared):
        print(f"\n🔬 Analyzing files: launching {len(shared['plan']['files'])} parallel analyses ...")
        shared["analyze_start"] = time.time()
        
        # Create file analysis tasks
        tasks = []
        for filename in shared['plan']['files']:
            tasks.append({
                'name': filename,
                'purpose': shared['plan']['file_purposes'][filename],
                'dependencies': shared['plan']['dependencies'].get(filename, []),
                'paper_content': shared['paper_content']
            })
        return tasks
    
    async def exec_async(self, file_info):
        print(f"🔍 Analyzing: {file_info['name']}")
        
        # Extract libraries mentioned in file purpose
        libraries = self.extract_libraries(file_info['purpose'])
        
        # Fetch relevant docs using context7
        docs = []
        for lib in libraries:
            try:
                lib_id = await resolve_library_id(lib)
                doc = await get_library_docs(
                    lib_id, 
                    topic=file_info['purpose'],
                    tokens=2000
                )
                docs.append(doc)
            except:
                pass  # Skip if library not found
        
        # Generate detailed logic spec
        prompt = load_yaml_field('prompts/analyze_file.md', 'prompt').format(
            filename=file_info['name'],
            purpose=file_info['purpose'],
            dependencies=file_info['dependencies'],
            library_docs="\n".join(docs),
            paper=file_info['paper_content']
        )
        
        spec = await call_llm_async(prompt)
        
        print(f"✅ {file_info['name']} analysis completed")
        return {
            'file': file_info['name'],
            'spec': spec,
            'libraries_used': libraries
        }
```

### 3. CodeGeneration Node

```python
class CodeGeneration(AsyncParallelBatchNode):
    """Generate code for each file in parallel"""
    
    async def prep_async(self, shared):
        print(f"\n💻 Code Generation: launching {len(shared['file_specs'])} parallel generations ...")
        shared["codegen_start"] = time.time()
        shared["generated_code"] = {}  # Store completed code
        
        # Prepare tasks with specs
        tasks = []
        for spec in shared['file_specs']:
            tasks.append({
                'filename': spec['file'],
                'spec': spec['spec'],
                'dependencies': shared['plan']['dependencies'].get(spec['file'], []),
                'interfaces': shared['plan']['shared_interfaces']
            })
        return tasks
    
    async def exec_async(self, task):
        print(f"🚀 Generating code for: {task['filename']}")
        
        # Wait for dependencies to complete
        dep_code = {}
        for dep in task['dependencies']:
            while dep not in self.shared['generated_code']:
                await asyncio.sleep(0.1)  # Simple polling
            dep_code[dep] = self.shared['generated_code'][dep]
        
        # Build prompt with spec + dependency code
        prompt = load_yaml_field('prompts/generate_code.md', 'prompt').format(
            filename=task['filename'],
            spec=task['spec'],
            interfaces=task['interfaces'],
            dependency_code=dep_code
        )
        
        code = await call_llm_async(prompt)
        
        # Store in shared state for dependents
        self.shared['generated_code'][task['filename']] = code
        
        print(f"✅ {task['filename']} completed")
        return {
            'filename': task['filename'],
            'code': code
        }
    
    async def post_async(self, shared, prep_res, exec_res):
        # Save all generated code to output directory
        output_dir = shared['output_dir'] / 'generated_repo'
        output_dir.mkdir(exist_ok=True)
        
        for result in exec_res:
            filepath = output_dir / result['filename']
            filepath.write_text(result['code'])
        
        duration = time.time() - shared["codegen_start"]
        print(f"✅ Code Generation: completed {len(exec_res)} files in {duration:.1f}s")
```

## Workflow Example

### Input Paper: "Implementing Fast Fourier Transform"

1. **BuildOverview** outputs:
   ```python
   {
       "nodes": ["FFT Algorithm", "Butterfly Operation", "Bit Reversal"],
       "algorithm_name": "Cooley-Tukey FFT",
       "mermaid_diagram": "..."
   }
   ```

2. **PlanGeneration** creates:
   ```python
   {
       "files": ["fft.py", "butterfly.py", "utils.py", "main.py", "test_fft.py"],
       "dependencies": {
           "main.py": ["fft.py"],
           "fft.py": ["butterfly.py", "utils.py"],
           "test_fft.py": ["fft.py"]
       },
       "file_purposes": {
           "fft.py": "Main FFT implementation using Cooley-Tukey algorithm",
           "butterfly.py": "Butterfly operation for FFT computation",
           "utils.py": "Bit reversal and helper functions",
           "main.py": "Entry point and usage example",
           "test_fft.py": "Unit tests for FFT implementation"
       }
   }
   ```

3. **AnalyzeFiles** (parallel execution):
   - `fft.py`: Fetches numpy.fft and scipy.fft docs
   - `butterfly.py`: Gets complex number operations docs
   - `utils.py`: No external library needed
   - All files analyzed simultaneously

4. **CodeGeneration** (parallel with dependency awareness):
   - First wave: `utils.py`, `butterfly.py` (no dependencies)
   - Second wave: `fft.py` (waits for utils + butterfly)
   - Third wave: `main.py`, `test_fft.py` (wait for fft)

### Timing Comparison

**Sequential (Paper2Code style)**:
- Planning: 10s
- Analyze 5 files: 5 × 10s = 50s
- Generate 5 files: 5 × 15s = 75s
- Total: 135s

**Our Parallel Approach**:
- BuildOverview: 15s
- PlanGeneration: 10s
- AnalyzeFiles: 10s (all parallel)
- CodeGeneration: 30s (dependency waves)
- Total: 65s

## Flow Configuration

```python
def create_paper2code_flow():
    """Create the complete paper to code generation flow"""
    flow = AsyncFlow()
    
    # Stage 1: Overview
    flow.add_node(BuildOverview())
    
    # Stage 2: Planning
    flow.add_node(PlanGeneration())
    
    # Stage 3: Analysis (parallel)
    flow.add_node(AnalyzeFiles())
    
    # Stage 4: Code Generation (parallel with deps)
    flow.add_node(CodeGeneration())
    
    return flow
```

## Multi-Language Support

Our system automatically detects the target language from the paper content:

```python
def detect_language(paper_content, algorithm_type):
    """Detect programming language from paper context"""
    # Language hints in paper
    if "CUDA" in paper_content or "GPU" in algorithm_type:
        return "cuda"
    elif "embedded" in paper_content or "real-time" in algorithm_type:
        return "cpp"
    elif "web" in paper_content:
        return "javascript"
    else:
        return "python"  # Default
```

Each language gets appropriate:
- File extensions (.py, .cpp, .cu, .js)
- Package managers (pip, cmake, npm)
- Library documentation (via context7)
- Code style conventions

## MVP Implementation Steps

1. Create prompt templates:
   - `prompts/plan_generation.md`
   - `prompts/analyze_file.md`
   - `prompts/generate_code.md`

2. Implement nodes in `src/nodes.py`:
   - Add PlanGeneration class
   - Extend with AnalyzeFiles
   - Add CodeGeneration

3. Update flow in `src/paper2code.py`:
   - Add new nodes to pipeline
   - Test with sample papers

4. Integrate context7 MCP:
   - Add to AnalyzeFiles.exec_async
   - Handle library resolution

5. Test with different paper types:
   - ML/AI papers → Python
   - Systems papers → C++/Rust
   - Graphics papers → CUDA/GLSL
   - Web papers → JavaScript/TypeScript