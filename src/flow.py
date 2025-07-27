"""Paper2Code flow using PocketFlow"""

from pocketflow import AsyncFlow
from nodes import BuildOverview, ProcessNode


def create_paper2code_flow():
    """Creates the 2-node Paper2Code pipeline"""
    
    # Instantiate nodes with retry configuration
    build = BuildOverview(max_retries=3, wait=10)
    process = ProcessNode(max_retries=3, wait=10)  # AsyncParallelBatchNode - runs N parallel pipelines
    
    # Chain nodes
    build >> process
    
    # Create async flow since we have async nodes
    return AsyncFlow(start=build)