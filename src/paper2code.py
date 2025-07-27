#!/usr/bin/env python3
"""
Paper2Code: Extract algorithms from papers using AI
"""

import sys
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import the flow
from flow import create_paper2code_flow

# Paths
BASE_DIR = Path(__file__).parent.parent


async def main():
    """Main entry point using PocketFlow"""
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
    
    # Create shared state
    shared = {
        "paper_content": paper_content,
        "output_dir": output_dir,
        "paper_name": paper_name
    }
    
    # Create and run async flow
    print("\n🚀 Running Paper2Code Pipeline...")
    flow = create_paper2code_flow()
    await flow.run_async(shared)
    
    # Summary
    total_time = time.time() - start_time
    print("\n✅ Pipeline completed successfully!")
    print(f"\n📊 Results:")
    print(f"  - Algorithm overview saved")
    print(f"  - Researched {len(shared['researched_nodes'])} nodes")
    
    print(f"\n⏱️  Total time: {total_time:.1f}s")
    print(f"📁 All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())