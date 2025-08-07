#!/usr/bin/env python3
"""
Generate static network graph JSON for the frontend.
This script queries all chapter similarities and generates a JSON file
that can be imported directly by the frontend at build time.
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Any, Set
from pathlib import Path
from collections import defaultdict

# Add the scripts directory to the path so we can import database
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager, Chapter, Video, ChapterSimilarity
from clustering import hierarchical_clustering, get_cluster_statistics

def generate_network_graph_json(
    similarity_threshold: float = 0.5,
    output_path: str = "../public/network-graph.json",
    min_component_size: int = 4,
    max_cluster_size: int = 10
) -> Dict[str, Any]:
    """
    Generate network graph JSON data.
    
    Args:
        similarity_threshold: Minimum similarity score to include (0.0 to 1.0)
        output_path: Path to save the JSON file
    
    Returns:
        Dictionary containing the network graph data
    """
    print(f"Generating network graph with similarity threshold: {similarity_threshold}")
    
    # Initialize database connection
    db_manager = DatabaseManager()
    
    try:
        with db_manager.get_session() as session:
            # Get all similarities above threshold, excluding self-loops
            similarities_query = session.query(
                ChapterSimilarity.source_chapter_id,
                ChapterSimilarity.dest_chapter_id,
                ChapterSimilarity.similarity_score
            ).filter(
                ChapterSimilarity.similarity_score >= similarity_threshold,
                ChapterSimilarity.source_chapter_id != ChapterSimilarity.dest_chapter_id
            ).order_by(ChapterSimilarity.similarity_score.desc())
            
            similarities = similarities_query.all()
            print(f"Found {len(similarities)} similarities above threshold")
            
            # Get unique chapter IDs from similarities
            chapter_ids: Set[int] = set()
            for sim in similarities:
                chapter_ids.add(sim.source_chapter_id)
                chapter_ids.add(sim.dest_chapter_id)
            
            print(f"Found {len(chapter_ids)} unique chapters")
            
            # Get chapter details with video information
            chapters_query = session.query(
                Chapter.id,
                Chapter.video_id,
                Chapter.chapter_idx,
                Chapter.chapter_name,
                Chapter.chapter_summary,
                Chapter.start_time,
                Chapter.end_time,
                Video.youtube_id,
                Video.title.label('video_title')
            ).join(
                Video, Chapter.video_id == Video.id
            ).filter(
                Chapter.id.in_(chapter_ids)
            )
            
            chapters = chapters_query.all()
            print(f"Retrieved {len(chapters)} chapter details")
            
            # Convert to Cytoscape format
            nodes = []
            node_ids = []
            for chapter in chapters:
                node_id = str(chapter.id)
                nodes.append({
                    "data": {
                        "id": node_id,
                        "label": chapter.chapter_name,
                    }
                })
                node_ids.append(node_id)
            
            edges = []
            edge_tuples = []
            for similarity in similarities:
                source_id = str(similarity.source_chapter_id)
                target_id = str(similarity.dest_chapter_id)
                edges.append({
                    "data": {
                        "id": f"{source_id}-{target_id}",
                        "source": source_id,
                        "target": target_id,
                        "similarityScore": similarity.similarity_score,
                    }
                })
                edge_tuples.append((source_id, target_id, similarity.similarity_score))
            
            # Use hierarchical clustering
            print(f"Using hierarchical clustering with max cluster size: {max_cluster_size}")
            communities = hierarchical_clustering(node_ids, edge_tuples, max_cluster_size=max_cluster_size)
            
            # Get cluster statistics
            cluster_stats = get_cluster_statistics(communities)
            print(f"Found {cluster_stats['total_clusters']} communities")
            print(f"Cluster sizes (top 10): {cluster_stats['cluster_sizes'][:10]}")
            print(f"Largest cluster: {cluster_stats['largest_cluster']} nodes")
            print(f"Average cluster size: {cluster_stats['average_cluster_size']:.1f} nodes")
            
            # Filter out small clusters
            valid_nodes = set()
            cluster_mapping = {}
            cluster_info = []
            
            for node_id, cluster_id in communities.items():
                cluster_size = sum(1 for n, c in communities.items() if c == cluster_id)
                if cluster_size >= min_component_size:
                    valid_nodes.add(node_id)
                    cluster_mapping[node_id] = cluster_id
            
            # Create cluster info for valid clusters
            valid_clusters = {}
            for node_id, cluster_id in communities.items():
                if node_id in valid_nodes:
                    if cluster_id not in valid_clusters:
                        valid_clusters[cluster_id] = []
                    valid_clusters[cluster_id].append(node_id)
            
            for cluster_id, node_list in valid_clusters.items():
                cluster_info.append({
                    "clusterId": cluster_id,
                    "size": len(node_list),
                    "nodeIds": node_list
                })
            
            print(f"After filtering: {len(valid_nodes)} nodes in clusters with {min_component_size}+ nodes")
            print(f"Valid clusters: {len(valid_clusters)}")
            
            # Filter nodes and edges to only include valid components, and add cluster IDs
            filtered_nodes = []
            for node in nodes:
                if node["data"]["id"] in valid_nodes:
                    node_data = node["data"].copy()
                    node_data["clusterId"] = cluster_mapping[node["data"]["id"]]
                    filtered_nodes.append({"data": node_data})
            
            filtered_edges = [edge for edge in edges if edge["data"]["source"] in valid_nodes and edge["data"]["target"] in valid_nodes]
            
            # Count edges being removed
            removed_edges = len(edges) - len(filtered_edges)
            print(f"Removing {removed_edges} edges from small components")
            
            # Update stats
            total_nodes = len(filtered_nodes)
            total_edges = len(filtered_edges)
            average_similarity = sum(edge["data"]["similarityScore"] for edge in filtered_edges) / total_edges if total_edges > 0 else 0
            
            print(f"Cluster sizes: {[len(node_list) for node_list in valid_clusters.values()]}")
            
            network_data = {
                "nodes": filtered_nodes,
                "edges": filtered_edges,
                "clusters": cluster_info,
                "stats": {
                    "totalNodes": total_nodes,
                    "totalEdges": total_edges,
                    "totalClusters": len(valid_clusters),
                    "averageSimilarity": round(average_similarity, 6),
                },
                "metadata": {
                    "generatedAt": datetime.now().isoformat(),
                    "similarityThreshold": similarity_threshold,
                    "version": "1.0"
                }
            }
            
            # Save to JSON file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(network_data, f, indent=2, ensure_ascii=False)
            
            print(f"Network graph JSON saved to: {output_file}")
            print(f"Stats: {total_nodes} nodes, {total_edges} edges, avg similarity: {average_similarity:.6f}")
            
            return network_data
            
    except Exception as e:
        print(f"Error generating network graph: {e}")
        raise

def main():
    """Main function to run the network graph generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate network graph JSON")
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5,
        help="Similarity threshold (0.0 to 1.0, default: 0.5)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./public/network-graph.json",
        help="Output JSON file path (default: ./public/network-graph.json)"
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=4,
        help="Minimum number of nodes required in a cluster (default: 4)"
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=10,
        help="Maximum cluster size (default: 10)"
    )
    
    args = parser.parse_args()
    
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: Similarity threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    try:
        generate_network_graph_json(args.threshold, args.output, args.min_component_size, args.max_cluster_size)
        print("Network graph generation completed successfully!")
    except Exception as e:
        print(f"Failed to generate network graph: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 