"""Paper2Code flow using PocketFlow"""

from pocketflow import AsyncFlow
from nodes import IdentifyNode, AnalyzeNode, QueryNode, ResearchNode


def create_paper2code_flow():
    """Creates the 4-node Paper2Code pipeline"""
    
    # Instantiate nodes
    identify = IdentifyNode()
    analyze = AnalyzeNode()
    query = QueryNode()
    research = ResearchNode()  # AsyncParallelBatchNode for parallel execution
    
    # Chain nodes
    identify >> analyze >> query >> research
    
    # Create async flow since we have async nodes
    return AsyncFlow(start=identify)