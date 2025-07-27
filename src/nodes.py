"""All nodes for Paper2Code pipeline"""

import os
import re
import yaml
import asyncio
import aiohttp
import time
from pathlib import Path
from openai import OpenAI
from pocketflow import Node, AsyncParallelBatchNode

import config

# Paths
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
PROMPTS_DIR = BASE_DIR / "prompts"


def load_yaml_field(filepath: Path, field: str) -> str:
    """Load specific field from YAML file"""
    with open(filepath) as f:
        return yaml.safe_load(f)[field]


async def call_openai(prompt: str, model: str = config.OPENAI_MODEL) -> str:
    """Make OpenAI API call"""
    client = OpenAI(api_key=os.getenv(config.OPENAI_API_KEY))
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


class BuildOverview(Node):
    """Build algorithm overview incrementally through 3 focused steps"""
    
    def prep(self, shared):
        print("\n📚 BuildOverview: Starting 3-step overview generation...")
        shared["build_start"] = time.time()
        return shared["paper_content"]
    
    def exec(self, paper_content):
        # Step 1: Generate Math Model
        print("  📐 Step 1: Generating mathematical model...")
        math_template = load_yaml_field(TEMPLATES_DIR / 'math_model.md', 'math_model_template')
        math_prompt = load_yaml_field(PROMPTS_DIR / 'generate_math_model.md', 'prompt').format(
            paper_content=paper_content,
            math_model_template=math_template
        )
        
        client = OpenAI(api_key=os.getenv(config.OPENAI_API_KEY))
        math_response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": math_prompt}]
        )
        math_content = math_response.choices[0].message.content
        
        # Extract nodes list from response
        nodes_match = re.search(r'NODES:\s*\n(.*?)$', math_content, re.DOTALL)
        if nodes_match:
            nodes_text = nodes_match.group(1).strip()
            # Extract node names from bullet list
            nodes_list = []
            for line in nodes_text.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    nodes_list.append(line[2:].strip())
            math_overview = math_content[:nodes_match.start()].strip()
        else:
            raise ValueError("Failed to extract nodes list from math model response")
        
        # Extract algorithm name from math overview
        algorithm_name = math_overview.split('\n')[0].replace('##', '').strip()
        
        # Step 2: Generate Pipeline
        print("  🔗 Step 2: Generating pipeline and data flow...")
        pipeline_template = load_yaml_field(TEMPLATES_DIR / 'pipeline.md', 'pipeline_template')
        pipeline_prompt = load_yaml_field(PROMPTS_DIR / 'generate_pipeline.md', 'prompt').format(
            algorithm_name=algorithm_name,
            nodes_list=', '.join(nodes_list),
            paper_content=paper_content,
            pipeline_template=pipeline_template
        )
        
        pipeline_response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": pipeline_prompt}]
        )
        pipeline_overview = pipeline_response.choices[0].message.content
        
        # Step 3: Generate Node Details
        print("  📝 Step 3: Generating detailed node descriptions...")
        node_details_template = load_yaml_field(TEMPLATES_DIR / 'node_details.md', 'node_details_template')
        node_details_prompt = load_yaml_field(PROMPTS_DIR / 'generate_node_details.md', 'prompt').format(
            algorithm_name=algorithm_name,
            nodes_list=', '.join(nodes_list),
            math_overview=math_overview,
            pipeline_overview=pipeline_overview,
            paper_content=paper_content,
            node_details_template=node_details_template
        )
        
        node_details_response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": node_details_prompt}]
        )
        node_details = node_details_response.choices[0].message.content
        
        return {
            "math_overview": math_overview,
            "pipeline_overview": pipeline_overview,
            "node_details": node_details,
            "nodes_list": nodes_list,
            "algorithm_name": algorithm_name
        }
    
    def post(self, shared, prep_res, exec_res):
        # Assemble final algorithm overview
        output_dir = shared["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all sections
        final_overview = f"{exec_res['math_overview']}\n\n{exec_res['pipeline_overview']}\n\n{exec_res['node_details']}"
        
        # Save algorithm overview
        (output_dir / "algorithm_overview.md").write_text(final_overview)
        
        # Store nodes for ProcessNode
        shared["nodes"] = [{"name": name, "description": ""} for name in exec_res['nodes_list']]
        shared["algorithm_name"] = exec_res['algorithm_name']
        
        duration = time.time() - shared["build_start"]
        print(f"✅ BuildOverview: Completed in {duration:.1f}s with {len(exec_res['nodes_list'])} nodes")



class ProcessNode(AsyncParallelBatchNode):
    """Process all nodes in parallel - each gets its own query->research pipeline"""
    
    async def prep_async(self, shared):
        print(f"\n🚀 ProcessNode: Launching {len(shared['nodes'])} parallel pipelines...")
        shared["process_start"] = time.time()
        
        # Return list of nodes to process
        return shared["nodes"]
    
    async def exec_async(self, node):
        """Execute the query->research pipeline for one node"""
        # Step 1: Generate query
        print(f"   🔍 Generating query for: {node['name']}")
        
        query_template = load_yaml_field(TEMPLATES_DIR / 'node_query.md', 'node_query_template')
        prompt = load_yaml_field(PROMPTS_DIR / 'query.md', 'prompt').format(
            node_name=node['name'],
            node_description=node['description'],
            query_template=query_template
        )
        
        query_text = await call_openai(prompt)
        if "Query: |" in query_text:
            query_text = query_text.split("Query: |")[1].strip()
        
        # Step 2: Research query
        print(f"   🚀 Started researching: {node['name']}")
        start_time = time.time()
        
        step_template = load_yaml_field(TEMPLATES_DIR / 'algorithm_step.md', 'algorithm_step_template')
        research_prompt = load_yaml_field(PROMPTS_DIR / 'research.md', 'prompt').format(
            query=query_text,
            step_template=step_template
        )
        
        headers = {
            "Authorization": f"Bearer {os.getenv(config.PERPLEXITY_API_KEY)}",
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
                safe_name = result['name'].replace('/', '-').replace('\\', '-').replace(':', '-')
                (output_dir / f"{safe_name}.md").write_text(result['content'])
                saved_nodes.append(result['name'])
        
        shared["researched_nodes"] = saved_nodes
        duration = time.time() - shared["process_start"]
        print(f"✅ ProcessNode: Completed! Processed {len(saved_nodes)} nodes in {duration:.1f}s")