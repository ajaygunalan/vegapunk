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
from pocketflow import Node, BatchNode, AsyncParallelBatchNode, AsyncFlow

# Paths
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
PROMPTS_DIR = BASE_DIR / "prompts"


def load_yaml_field(filepath: Path, field: str) -> str:
    """Load specific field from YAML file"""
    with open(filepath) as f:
        return yaml.safe_load(f)[field]


class IdentifyNode(Node):
    """Find all computational nodes in paper"""
    
    def prep(self, shared):
        print("\n🔍 IdentifyNode: Starting...")
        shared["identify_start"] = time.time()
        return shared["paper_content"]
    
    def exec(self, paper_content):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load prompt from file
        prompt = load_yaml_field(PROMPTS_DIR / 'identify.md', 'prompt').format(
            paper_content=paper_content
        )
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def post(self, shared, prep_res, exec_res):
        shared["nodes"] = exec_res["nodes"]
        shared["summary"] = exec_res["summary"]
        duration = time.time() - shared["identify_start"]
        print(f"✅ IdentifyNode: Found {len(exec_res['nodes'])} nodes in {duration:.1f}s")


class AnalyzeNode(Node):
    """Create algorithm overview with relationships"""
    
    def prep(self, shared):
        print("\n📊 AnalyzeNode: Creating algorithm overview...")
        shared["analyze_start"] = time.time()
        return {
            "nodes": shared["nodes"],
            "summary": shared["summary"],
            "paper_content": shared["paper_content"]
        }
    
    def exec(self, prep_res):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load templates
        overview_template = load_yaml_field(TEMPLATES_DIR / 'algorithm_overview.md', 'algorithm_overview_template')
        
        # Format nodes for prompt
        nodes_text = "\n".join([f"- {n['name']}: {n['description']}" for n in prep_res['nodes']])
        
        # Load and format prompt
        prompt = load_yaml_field(PROMPTS_DIR / 'analyze.md', 'prompt').format(
            nodes_text=nodes_text,
            summary=prep_res['summary'],
            paper_content=prep_res['paper_content'],
            overview_template=overview_template
        )
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def post(self, shared, prep_res, exec_res):
        # Save algorithm overview
        output_dir = shared["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract just the overview part (before NODES:)
        overview = exec_res.split("NODES:")[0].strip()
        if overview.startswith("algorithm_overview_template: |"):
            overview = overview.replace("algorithm_overview_template: |", "").strip()
        
        (output_dir / "algorithm_overview.md").write_text(overview)
        shared["raw_analyze_output"] = exec_res  # Keep full output for debugging
        duration = time.time() - shared["analyze_start"]
        print(f"✅ AnalyzeNode: Overview saved in {duration:.1f}s")


class QueryNode(Node):
    """Generate research questions for each node"""
    
    def prep(self, shared):
        nodes = shared["nodes"]
        print(f"\n❓ QueryNode: Generating queries for {len(nodes)} nodes...")
        shared["query_start"] = time.time()
        return nodes
    
    def exec(self, nodes):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load query template
        query_template = load_yaml_field(TEMPLATES_DIR / 'node_query.md', 'node_query_template')
        
        queries = {}
        for i, node in enumerate(nodes, 1):
            print(f"   Generating query {i}/{len(nodes)}: {node['name']}")
            # Load and format prompt
            prompt = load_yaml_field(PROMPTS_DIR / 'query.md', 'prompt').format(
                node_name=node['name'],
                node_description=node['description'],
                query_template=query_template
            )
            
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract query text
            query_text = response.choices[0].message.content
            if "Query: |" in query_text:
                query_text = query_text.split("Query: |")[1].strip()
            
            queries[node['name']] = query_text
        
        return queries
    
    def post(self, shared, prep_res, exec_res):
        shared["queries"] = exec_res
        duration = time.time() - shared["query_start"]
        print(f"✅ QueryNode: Generated {len(exec_res)} queries in {duration:.1f}s")


class ResearchNode(AsyncParallelBatchNode):
    """Research nodes in parallel using Perplexity"""
    
    async def prep_async(self, shared):
        print(f"\n🔬 ResearchNode: Starting parallel research...")
        shared["research_start"] = time.time()
        # Convert queries dict to list for AsyncParallelBatchNode
        items = []
        for name, query in shared["queries"].items():
            items.append({"name": name, "query": query})
        return items
    
    async def exec_async(self, item):
        """Research single node asynchronously"""
        name = item["name"]
        query = item["query"]
        
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
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'name': name,
                        'content': data['choices'][0]['message']['content']
                    }
                else:
                    print(f"Error researching {name}: {response.status}")
                    return None
    
    async def post_async(self, shared, prep_res, exec_res_list):
        # Save results directly
        output_dir = shared["output_dir"]
        saved_nodes = []
        
        for result in exec_res_list:
            if result and not isinstance(result, Exception):
                (output_dir / f"{result['name']}.md").write_text(result['content'])
                saved_nodes.append(result['name'])
        
        shared["researched_nodes"] = saved_nodes
        duration = time.time() - shared["research_start"]
        print(f"✅ ResearchNode: Completed! Researched {len(saved_nodes)} nodes in parallel in {duration:.1f}s")