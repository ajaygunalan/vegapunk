"""All nodes for Paper2Code pipeline"""

import os
import re
import yaml
import aiohttp
import time
from pathlib import Path
from pocketflow import Node, AsyncParallelBatchNode

import config
from utils.llm import call_openai, call_openai_async


def load_yaml_field(filepath: Path, field: str) -> str:
    """Load specific field from YAML file.
    
    Args:
        filepath: Path to YAML file
        field: Field name to extract
        
    Returns:
        The value of the specified field
    """
    with open(filepath) as f:
        return yaml.safe_load(f)[field]


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a filename.
    
    Args:
        name: The string to sanitize
        
    Returns:
        A sanitized string safe for use as a filename
    """
    return name.replace('/', '-').replace('\\', '-').replace(':', '-')


class BuildOverview(Node):
    """Generate complete algorithm overview in a single API call.
    
    This node:
    1. Loads the algorithm overview template
    2. Calls OpenAI to generate a complete overview with Mermaid diagram
    3. Extracts and validates the nodes list and algorithm name
    4. Saves the overview to output directory
    """
    
    def prep(self, shared):
        print("\n📚 BuildOverview: Generating complete algorithm overview...")
        shared["build_start"] = time.time()
        return shared["paper_content"]
    
    def exec(self, paper_content):
        # Load template and prompt
        template = load_yaml_field(config.TEMPLATES_DIR / 'algorithm_overview_v2.md', 'algorithm_overview_template')
        prompt = load_yaml_field(config.PROMPTS_DIR / 'generate_overview.md', 'prompt').format(
            paper_content=paper_content,
            template=template
        )
        
        # Single API call using centralized function
        content = call_openai(prompt, max_tokens=4000)
        
        # Validate response is not empty
        if not content or not content.strip():
            raise ValueError("LLM returned empty response")
        
        # Extract nodes from new format: 1. [[Node Name]] - description
        nodes_list = re.findall(r'^\d+\.\s*\[\[(.*?)\]\]', content, re.MULTILINE)
        
        # Validate we found nodes
        if not nodes_list:
            raise ValueError("Failed to extract nodes from LLM response. Expected format: '1. [[Node Name]]'")
        
        # Extract algorithm name
        lines = content.split('\n')
        algorithm_name = lines[0].replace('##', '').strip() if lines else "Unknown Algorithm"
        
        # Validate algorithm name
        if not algorithm_name or algorithm_name == "Unknown Algorithm":
            raise ValueError("Failed to extract algorithm name from response")
        
        return {
            "content": content,
            "nodes_list": nodes_list,
            "algorithm_name": algorithm_name
        }
    
    def post(self, shared, prep_res, exec_res):
        # Save complete algorithm overview
        output_dir = shared["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the complete content
        (output_dir / "algorithm_overview.md").write_text(exec_res['content'])
        
        # Store nodes for ProcessNode
        shared["nodes"] = [{"name": name, "description": ""} for name in exec_res['nodes_list']]
        shared["algorithm_name"] = exec_res['algorithm_name']
        
        duration = time.time() - shared["build_start"]
        print(f"✅ BuildOverview: Completed in {duration:.1f}s with {len(exec_res['nodes_list'])} nodes")



class ProcessNode(AsyncParallelBatchNode):
    """Process all nodes in parallel - each gets its own query->research pipeline.
    
    This node:
    1. Takes the list of nodes from BuildOverview
    2. For each node in parallel:
       - Generates a research query using OpenAI
       - Researches the query using Perplexity API
    3. Saves each researched node as a separate markdown file
    """
    
    async def prep_async(self, shared):
        print(f"\n🚀 ProcessNode: Launching {len(shared['nodes'])} parallel pipelines...")
        shared["process_start"] = time.time()
        
        # Return list of nodes to process
        return shared["nodes"]
    
    async def exec_async(self, node):
        """Execute the query->research pipeline for one node"""
        # Step 1: Generate query
        print(f"   🔍 Generating query for: {node['name']}")
        
        query_template = load_yaml_field(config.TEMPLATES_DIR / 'node_query.md', 'node_query_template')
        prompt = load_yaml_field(config.PROMPTS_DIR / 'query.md', 'prompt').format(
            node_name=node['name'],
            node_description=node['description'],
            query_template=query_template
        )
        
        # Generate query using async centralized function for parallel execution
        query_text = await call_openai_async(prompt)
        if "Query: |" in query_text:
            query_text = query_text.split("Query: |")[1].strip()
        
        # Step 2: Research query
        print(f"   🚀 Started researching: {node['name']}")
        start_time = time.time()
        
        step_template = load_yaml_field(config.TEMPLATES_DIR / 'algorithm_step.md', 'algorithm_step_template')
        research_prompt = load_yaml_field(config.PROMPTS_DIR / 'research.md', 'prompt').format(
            query=query_text,
            step_template=step_template
        )
        
        headers = {
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.PERPLEXITY_MODEL,
            "messages": [{"role": "user", "content": research_prompt}]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config.PERPLEXITY_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.PERPLEXITY_TIMEOUT)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    duration = time.time() - start_time
                    print(f"   ✅ Completed: {node['name']} ({duration:.1f}s)")
                    return {
                        'name': node['name'],
                        'content': data['choices'][0]['message']['content']
                    }
                else:
                    print(f"   ❌ Error researching {node['name']}: {response.status}")
                    return None
    
    async def post_async(self, shared, prep_res, exec_res_list):
        # Collect and save all results
        saved_nodes = []
        output_dir = shared["output_dir"]
        
        # exec_res_list contains results from each exec_async call
        for result in exec_res_list:
            if result:
                # Sanitize filename
                safe_name = sanitize_filename(result['name'])
                (output_dir / f"{safe_name}.md").write_text(result['content'])
                saved_nodes.append(result['name'])
        
        shared["researched_nodes"] = saved_nodes
        duration = time.time() - shared["process_start"]
        print(f"✅ ProcessNode: Completed! Processed {len(saved_nodes)} nodes in {duration:.1f}s")