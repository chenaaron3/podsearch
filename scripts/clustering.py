#!/usr/bin/env python3
"""
Hierarchical clustering for network graph analysis.
"""

from collections import defaultdict
from typing import Dict, List, Any, Tuple, Union
import networkx as nx
import json
from community import community_louvain

def hierarchical_clustering(nodes: List[str], edges: List[Tuple[str, str, float]], 
                          max_cluster_size: int = 10, resolution: float = 1.0,
                          return_tree: bool = False) -> Union[Dict[str, int], Dict[str, Any]]:
    """
    Recursively break down clusters until all are under max_cluster_size.
    
    Args:
        nodes: List of node IDs
        edges: List of (source, target, weight) tuples
        max_cluster_size: Maximum size for any cluster (default: 10)
        resolution: Resolution parameter for Louvain (higher = more communities)
        return_tree: If True, return full hierarchy tree; if False, return flat cluster mapping
    
    Returns:
        If return_tree=False: Dictionary mapping node_id to hierarchical_cluster_id
        If return_tree=True: Dictionary with tree, node_mapping, and cluster_details
    """
    # Create NetworkX graph
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    for source, target, weight in edges:
        G.add_edge(source, target, weight=weight)
    
    # Initialize tree structure if needed
    if return_tree:
        hierarchy_tree = {
            "id": "root",
            "name": "All Chapters",
            "size": len(nodes),
            "level": 0,
            "children": []
        }
        node_to_cluster = {}
        cluster_counter = 0
    else:
        # For flat clustering, just track cluster assignments
        final_clusters = {}
        next_cluster_id = 0
    
    def break_down_cluster(cluster_nodes: List[str], cluster_id: int, depth: int = 0, parent_node: Dict = None) -> Dict[str, int]:
        """Recursively break down a cluster if it's too large."""
        if return_tree:
            nonlocal cluster_counter
        else:
            nonlocal next_cluster_id
        
        if len(cluster_nodes) <= max_cluster_size:
            # Cluster is small enough
            if return_tree:
                # Create leaf node for tree structure
                leaf_node = {
                    "id": f"cluster_{cluster_counter}",
                    "name": f"Cluster {cluster_counter}",
                    "size": len(cluster_nodes),
                    "level": depth,
                    "nodes": cluster_nodes,
                    "cluster_id": cluster_counter,
                    "type": "leaf"
                }
                parent_node["children"].append(leaf_node)
                
                # Assign all nodes to this cluster
                for node in cluster_nodes:
                    node_to_cluster[node] = cluster_counter
                
                cluster_counter += 1
                return {node: cluster_counter - 1 for node in cluster_nodes}
            else:
                # For flat clustering, just return the mapping
                return {node: cluster_id for node in cluster_nodes}
        
        # Create subgraph for this cluster
        subgraph = G.subgraph(cluster_nodes)
        
        # Get edges within this cluster
        cluster_edges = [(u, v, d['weight']) for u, v, d in subgraph.edges(data=True)]
        
        if len(cluster_edges) == 0:
            # No edges within cluster, split arbitrarily
            if return_tree:
                # Create individual leaf nodes
                current_cluster_id = cluster_counter
                for i, node in enumerate(cluster_nodes):
                    node_to_cluster[node] = current_cluster_id + i
                    
                    leaf_node = {
                        "id": f"cluster_{current_cluster_id + i}",
                        "name": f"Cluster {current_cluster_id + i}",
                        "size": 1,
                        "level": depth + 1,
                        "nodes": [node],
                        "cluster_id": current_cluster_id + i,
                        "type": "leaf"
                    }
                    parent_node["children"].append(leaf_node)
                
                cluster_counter += len(cluster_nodes)
                return {node: current_cluster_id + i for i, node in enumerate(cluster_nodes)}
            else:
                # For flat clustering
                result = {}
                for i, node in enumerate(cluster_nodes):
                    result[node] = cluster_id + i
                next_cluster_id = max(next_cluster_id, cluster_id + len(cluster_nodes))
                return result
        
        # Run Louvain on the subgraph
        try:
            sub_communities = community_louvain.best_partition(subgraph, resolution=resolution)
            
            # Count nodes in each sub-community
            sub_community_sizes = defaultdict(list)
            for node, sub_comm_id in sub_communities.items():
                sub_community_sizes[sub_comm_id].append(node)
            
            # If we only got one community, try with higher resolution
            if len(sub_community_sizes) == 1:
                sub_communities = community_louvain.best_partition(subgraph, resolution=resolution * 2)
                sub_community_sizes = defaultdict(list)
                for node, sub_comm_id in sub_communities.items():
                    sub_community_sizes[sub_comm_id].append(node)
            
            # If still only one community, split arbitrarily
            if len(sub_community_sizes) == 1:
                if return_tree:
                    # Create individual leaf nodes
                    current_cluster_id = cluster_counter
                    for i, node in enumerate(cluster_nodes):
                        node_to_cluster[node] = current_cluster_id + i
                        
                        leaf_node = {
                            "id": f"cluster_{current_cluster_id + i}",
                            "name": f"Cluster {current_cluster_id + i}",
                            "size": 1,
                            "level": depth + 1,
                            "nodes": [node],
                            "cluster_id": current_cluster_id + i,
                            "type": "leaf"
                        }
                        parent_node["children"].append(leaf_node)
                    
                    cluster_counter += len(cluster_nodes)
                    return {node: current_cluster_id + i for i, node in enumerate(cluster_nodes)}
                else:
                    # For flat clustering
                    result = {}
                    for i, node in enumerate(cluster_nodes):
                        result[node] = cluster_id + i
                    next_cluster_id = max(next_cluster_id, cluster_id + len(cluster_nodes))
                    return result
            
            # Recursively process each sub-community
            if return_tree:
                # Create intermediate node for this level
                intermediate_node = {
                    "id": f"intermediate_{depth}_{len(parent_node['children'])}",
                    "name": f"Level {depth} Group",
                    "size": len(cluster_nodes),
                    "level": depth,
                    "children": [],
                    "type": "intermediate"
                }
                parent_node["children"].append(intermediate_node)
                
                # Process each sub-community
                result = {}
                for sub_comm_id, sub_nodes in sub_community_sizes.items():
                    sub_result = break_down_cluster(sub_nodes, cluster_counter, depth + 1, intermediate_node)
                    result.update(sub_result)
                return result
            else:
                # For flat clustering
                result = {}
                for sub_comm_id, sub_nodes in sub_community_sizes.items():
                    if len(sub_nodes) <= max_cluster_size:
                        # Sub-community is small enough
                        for node in sub_nodes:
                            result[node] = next_cluster_id
                        next_cluster_id += 1
                    else:
                        # Recursively break down this sub-community
                        sub_result = break_down_cluster(sub_nodes, next_cluster_id, depth + 1)
                        result.update(sub_result)
                        next_cluster_id = max(next_cluster_id, max(sub_result.values()) + 1)
                return result
            
        except Exception as e:
            print(f"Warning: Could not partition cluster of size {len(cluster_nodes)}: {e}")
            # Fallback: split arbitrarily
            if return_tree:
                # Create individual leaf nodes
                current_cluster_id = cluster_counter
                for i, node in enumerate(cluster_nodes):
                    node_to_cluster[node] = current_cluster_id + i
                    
                    leaf_node = {
                        "id": f"cluster_{current_cluster_id + i}",
                        "name": f"Cluster {current_cluster_id + i}",
                        "size": 1,
                        "level": depth + 1,
                        "nodes": [node],
                        "cluster_id": current_cluster_id + i,
                        "type": "leaf"
                    }
                    parent_node["children"].append(leaf_node)
                
                cluster_counter += len(cluster_nodes)
                return {node: current_cluster_id + i for i, node in enumerate(cluster_nodes)}
            else:
                # For flat clustering
                result = {}
                for i, node in enumerate(cluster_nodes):
                    result[node] = cluster_id + i
                next_cluster_id = max(next_cluster_id, cluster_id + len(cluster_nodes))
                return result
    
    # Start with initial clustering
    initial_communities = community_louvain.best_partition(G, resolution=resolution)
    
    if return_tree:
        # Group nodes by initial community
        initial_groups = defaultdict(list)
        for node, comm_id in initial_communities.items():
            initial_groups[comm_id].append(node)
        
        # Process each initial group
        for comm_id, group_nodes in initial_groups.items():
            break_down_cluster(group_nodes, cluster_counter, 1, hierarchy_tree)
        
        # Build cluster details
        cluster_details = {}
        for node, cluster_id in node_to_cluster.items():
            if cluster_id not in cluster_details:
                cluster_details[cluster_id] = {
                    "cluster_id": cluster_id,
                    "nodes": [],
                    "size": 0
                }
            cluster_details[cluster_id]["nodes"].append(node)
            cluster_details[cluster_id]["size"] += 1
        
        return {
            "tree": hierarchy_tree,
            "node_mapping": node_to_cluster,
            "cluster_details": cluster_details
        }
    else:
        # Process each initial cluster for flat clustering
        for node, initial_cluster_id in initial_communities.items():
            if node not in final_clusters:
                # Get all nodes in this initial cluster
                cluster_nodes = [n for n, c in initial_communities.items() if c == initial_cluster_id]
                
                # Break down this cluster
                cluster_result = break_down_cluster(cluster_nodes, next_cluster_id)
                final_clusters.update(cluster_result)
                next_cluster_id = max(next_cluster_id, max(cluster_result.values()) + 1)
        
        # Renumber clusters to be consecutive
        unique_clusters = sorted(set(final_clusters.values()))
        cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        
        return {node: cluster_mapping[cluster_id] for node, cluster_id in final_clusters.items()}

def get_cluster_statistics(communities: Dict[str, int]) -> Dict:
    """Get statistics about the clustering results."""
    cluster_sizes = defaultdict(int)
    for community_id in communities.values():
        cluster_sizes[community_id] += 1
    
    sizes = list(cluster_sizes.values())
    sizes.sort(reverse=True)
    
    return {
        "total_clusters": len(cluster_sizes),
        "cluster_sizes": sizes,
        "largest_cluster": max(sizes) if sizes else 0,
        "smallest_cluster": min(sizes) if sizes else 0,
        "average_cluster_size": sum(sizes) / len(sizes) if sizes else 0,
    }

def generate_hierarchy_json(nodes: List[str], edges: List[Tuple[str, str, float]], 
                          max_cluster_size: int = 10, output_path: str = "../public/hierarchy.json") -> Dict[str, Any]:
    """
    Generate a JSON file with hierarchical clustering data for visualization.
    """
    print("Building hierarchy tree...")
    hierarchy_data = hierarchical_clustering(nodes, edges, max_cluster_size, return_tree=True)
    
    # Add statistics
    cluster_sizes = [details["size"] for details in hierarchy_data["cluster_details"].values()]
    cluster_sizes.sort(reverse=True)
    
    hierarchy_data["statistics"] = {
        "total_clusters": len(hierarchy_data["cluster_details"]),
        "total_nodes": len(nodes),
        "cluster_sizes": cluster_sizes,
        "largest_cluster": max(cluster_sizes) if cluster_sizes else 0,
        "smallest_cluster": min(cluster_sizes) if cluster_sizes else 0,
        "average_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max_cluster_size
    }
    
    # Save to JSON
    import os
    from pathlib import Path
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hierarchy_data, f, indent=2, ensure_ascii=False)
    
    print(f"Hierarchy JSON saved to: {output_file}")
    print(f"Statistics: {hierarchy_data['statistics']}")
    
    return hierarchy_data 