"""All nodes for Paper2Code pipeline"""

import os
import re
import json
import yaml
import asyncio
import aiohttp
import time
from pathlib import Path
from openai import OpenAI
from pocketflow import Node, BatchNode, AsyncNode, AsyncParallelBatchNode, AsyncFlow

# Paths
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
PROMPTS_DIR = BASE_DIR / "prompts"


def load_yaml_field(filepath: Path, field: str) -> str:
    """Load specific field from YAML file"""
    with open(filepath) as f:
        return yaml.safe_load(f)[field]


class BuildOverview(Node):
    """Build algorithm overview incrementally through 3 focused steps"""
    
    def prep(self, shared):
        print("\n📚 BuildOverview: Starting 3-step overview generation...")
        shared["build_start"] = time.time()
        return shared["paper_content"]
    
    def exec(self, paper_content):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Step 1: Generate Math Model
        print("  📐 Step 1: Generating mathematical model...")
        math_template = load_yaml_field(TEMPLATES_DIR / 'math_model.md', 'math_model_template')
        math_prompt = load_yaml_field(PROMPTS_DIR / 'generate_math_model.md', 'prompt').format(
            paper_content=paper_content,
            math_model_template=math_template
        )
        
        math_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": math_prompt}]
        )
        math_content = math_response.choices[0].message.content
        
        # Extract nodes list from response
        nodes_match = re.search(r'\{"nodes":\s*\[(.*?)\]\}', math_content, re.DOTALL)
        if nodes_match:
            nodes_str = nodes_match.group(1)
            nodes_list = [n.strip().strip('"') for n in nodes_str.split(',')]
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
            model="gpt-4.1-mini",
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
            model="gpt-4.1-mini",
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


class GenerateQueryNode(AsyncNode):
    """Generate research question for one node"""
    
    async def prep_async(self, shared):
        # The node is passed as params from ProcessNode
        return self.params
    
    async def exec_async(self, node_data):
        # Extract node from params
        node = node_data['node']
        
        print(f"   🔍 Generating query for: {node['name']}")
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load query template
        query_template = load_yaml_field(TEMPLATES_DIR / 'node_query.md', 'node_query_template')
        
        # Load and format prompt
        prompt = load_yaml_field(PROMPTS_DIR / 'query.md', 'prompt').format(
            node_name=node['name'],
            node_description=node['description'],
            query_template=query_template
        )
        
        # Use async create (OpenAI SDK supports async)
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract query text
        query_text = response.choices[0].message.content
        if "Query: |" in query_text:
            query_text = query_text.split("Query: |")[1].strip()
        
        return query_text
    
    async def post_async(self, shared, prep_res, exec_res):
        # Store query and node info in shared for next node
        shared["query"] = exec_res
        shared["node"] = prep_res["node"]
        shared["output_dir"] = shared.get("output_dir")  # Pass through output_dir


class ResearchQueryNode(AsyncNode):
    """Research single query using Perplexity"""
    
    async def prep_async(self, shared):
        # Get data from previous node
        return {
            "name": shared["node"]["name"],
            "query": shared["query"]
        }
    
    async def exec_async(self, item):
        """Research single query"""
        name = item["name"]
        query = item["query"]
        
        print(f"   🚀 Started researching: {name}")
        start_time = time.time()
        
        # Load templates and prompts
        step_template = load_yaml_field(TEMPLATES_DIR / 'algorithm_step.md', 'algorithm_step_template')
        prompt = load_yaml_field(PROMPTS_DIR / 'research.md', 'prompt').format(
            query=query,
            step_template=step_template
        )
        
        headers = {
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "sonar",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    duration = time.time() - start_time
                    print(f"   ✅ Completed: {name} ({duration:.1f}s)")
                    return {
                        'name': name,
                        'content': data['choices'][0]['message']['content']
                    }
                else:
                    print(f"   ❌ Error researching {name}: {response.status}")
                    return None
    
    async def post_async(self, shared, prep_res, exec_res):
        # Save result for ProcessNode to collect
        if exec_res:
            shared["result"] = exec_res


class ProcessNode(AsyncParallelBatchNode):
    """Process all nodes in parallel - each gets its own query->research pipeline"""
    
    async def prep_async(self, shared):
        print(f"\n🚀 ProcessNode: Launching {len(shared['nodes'])} parallel pipelines...")
        shared["process_start"] = time.time()
        
        # Return list of nodes to process
        return shared["nodes"]
    
    async def exec_async(self, node):
        """Execute the query->research pipeline for one node"""
        # Create a fresh sub-flow for this node
        generate = GenerateQueryNode()
        research = ResearchQueryNode()
        generate >> research
        sub_flow = AsyncFlow(start=generate)
        
        # Create a shared context for this sub-flow
        sub_shared = {
            "node": node,
            "output_dir": None  # Will be passed through from parent
        }
        
        # Set params for the sub-flow
        sub_flow.set_params({"node": node})
        
        # Run the sub-flow
        await sub_flow.run_async(sub_shared)
        
        # Return the result from the sub-flow
        return sub_shared.get("result")
    
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