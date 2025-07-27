#!/usr/bin/env python3
"""
Paper2Code: Extract algorithms from papers using AI
"""

import os
import re
import sys
import yaml
import time
import asyncio
import aiohttp
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
PROMPTS_DIR = BASE_DIR / "prompts"


def load_yaml_field(filepath: Path, field: str) -> str:
    """Load specific field from YAML file"""
    with open(filepath) as f:
        return yaml.safe_load(f)[field]


class AnalyzeNode:
    """Extract algorithm structure using OpenAI"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def run(self, paper_content: str, output_dir: Path) -> str:
        """Extract algorithm nodes from paper"""
        system_prompt = load_yaml_field(PROMPTS_DIR / 'analyze_system_prompt.yml', 'system_prompt').format(
            overview_template=load_yaml_field(TEMPLATES_DIR / 'algorithm_overview.yml', 'algorithm_overview_template'),
            query_template=load_yaml_field(TEMPLATES_DIR / 'node_query.yml', 'node_query_template')
        )
        
        user_prompt = load_yaml_field(PROMPTS_DIR / 'analyze_user_prompt.yml', 'user_prompt').format(
            paper_content=paper_content
        )
        
        # Call OpenAI
        print("   Calling OpenAI API...")
        start = time.time()
        response = self.client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        elapsed = time.time() - start
        print(f"   ✓ Completed in {elapsed:.1f}s")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analyze_raw.md").write_text(response.choices[0].message.content)
        return response.choices[0].message.content


class ResearchNode:
    """Research algorithm nodes using Perplexity"""
    
    def __init__(self):
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        
    async def research_node(self, name: str, query: str, session: aiohttp.ClientSession) -> dict:
        """Research single node"""
        step_template = load_yaml_field(TEMPLATES_DIR / 'algorithm_step.yml', 'algorithm_step_template')
        system_prompt = load_yaml_field(PROMPTS_DIR / 'research_system_prompt.yml', 'system_prompt')
        user_prompt = load_yaml_field(PROMPTS_DIR / 'research_user_prompt.yml', 'user_prompt').format(
            query=query,
            step_template=step_template
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
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
    
    async def run(self, analyze_output: str, output_dir: Path) -> dict:
        """Parse nodes and research in parallel"""
        # Parse output
        parts = analyze_output.split("NODES:")
        if len(parts) < 2:
            return {'node_count': 0, 'nodes': []}
        
        # Save overview
        overview = parts[0].split("LINKS:")[0].strip()
        # Remove template header if present
        if overview.startswith("algorithm_overview_template: |"):
            overview = overview.replace("algorithm_overview_template: |", "").strip()
        (output_dir / "algorithm_overview.md").write_text(overview)
        
        # Extract nodes
        nodes = [
            {'name': name.strip(), 'query': q.group(1).strip() if (q := re.search(r'Query:\s*\|(.*?)(?=$)', content.strip(), re.DOTALL)) else content.strip()}
            for name, content in re.findall(r'===([^=]+)===(.*?)(?====|$)', parts[1], re.DOTALL)
        ]
        
        # Research nodes in parallel
        if not nodes:
            return {'node_count': 0, 'nodes': []}
        
        print(f"   Researching {len(nodes)} nodes in parallel...")
        start = time.time()
            
        async with aiohttp.ClientSession() as session:
            tasks = [self.research_node(n['name'], n['query'], session) for n in nodes]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Save results
            valid_results = [
                result['name'] 
                for result in results 
                if result and not isinstance(result, Exception)
                if (output_dir / f"{result['name']}.md").write_text(result['content']) or True
            ]
            
            elapsed = time.time() - start
            print(f"   ✓ Completed in {elapsed:.1f}s")
            
            return {
                'node_count': len(valid_results),
                'nodes': valid_results
            }


async def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python paper2code.py <paper_markdown_path>")
        sys.exit(1)
    
    paper_path = Path(sys.argv[1])
    if not paper_path.exists():
        print(f"Error: File '{paper_path}' not found")
        sys.exit(1)
    
    # Get paper name from path
    paper_name = next((paper_path.parts[i+1] for i, p in enumerate(paper_path.parts) 
                      if p == "test_samples" and i+1 < len(paper_path.parts)), paper_path.stem)
    
    output_dir = BASE_DIR / "output" / paper_name
    paper_content = paper_path.read_text()
    
    print(f"📚 Processing: {paper_path}")
    start_time = time.time()
    
    # Run analysis
    print("\n🚀 Running AnalyzeNode...")
    analyze = AnalyzeNode()
    analyze_output = analyze.run(paper_content, output_dir)
    
    # Run research
    print("\n🚀 Running ResearchNode...")
    research = ResearchNode()
    results = await research.run(analyze_output, output_dir)
    
    # Summary
    total_time = time.time() - start_time
    print("\n✅ Pipeline completed successfully!")
    print(f"\n📊 Results:")
    print(f"  - Algorithm overview saved")
    print(f"  - Researched {results['node_count']} nodes")
    if results['nodes']:
        print(f"  - Nodes: {', '.join(results['nodes'])}")
    print(f"\n⏱️  Total time: {total_time:.1f}s")
    print(f"📁 All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())