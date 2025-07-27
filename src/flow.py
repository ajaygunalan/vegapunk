"""Paper2Code flow using PocketFlow"""

from pocketflow import AsyncFlow
from nodes import BuildOverview, ProcessNode


def create_paper2code_flow():
    """Creates the 2-node Paper2Code pipeline"""
    
    # Instantiate nodes
    build = BuildOverview()
    process = ProcessNode()  # Runs N parallel pipelines
    
    # Chain nodes
    build >> process
    
    # Create async flow since we have async nodes
    return AsyncFlow(start=build)