#!/usr/bin/env python3
"""
Regenerate hierarchy and labeled hierarchy files.
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clustering import generate_hierarchy_json
from topic_labeling import generate_labeled_hierarchy

def main():
    """Regenerate hierarchy files."""
    
    # Load existing network graph data
    network_graph_path = Path("../public/network-graph.json")
    
    if not network_graph_path.exists():
        print("Error: network-graph.json not found. Please generate it first.")
        return
    
    print("Loading network graph data...")
    import json
    with open(network_graph_path, 'r', encoding='utf-8') as f:
        network_data = json.load(f)
    
    # Extract nodes and edges
    nodes = [node["data"]["id"] for node in network_data["nodes"]]
    edges = [(edge["data"]["source"], edge["data"]["target"], edge["data"]["similarityScore"]) 
             for edge in network_data["edges"]]
    
    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges")
    
    # Generate hierarchy data
    print("Generating hierarchy...")
    hierarchy_data = generate_hierarchy_json(
        nodes=nodes,
        edges=edges,
        max_cluster_size=10,
        output_path="../public/hierarchy.json"
    )
    
    # Generate labeled hierarchy
    print("Generating labeled hierarchy...")
    generate_labeled_hierarchy(
        hierarchy_path="../public/hierarchy.json",
        output_path="../public/labeled-hierarchy.json"
    )
    
    print("Hierarchy regeneration completed!")

if __name__ == "__main__":
    main() 