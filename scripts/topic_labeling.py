#!/usr/bin/env python3
"""
Topic labeling for hierarchical clustering.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import json
from pathlib import Path

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract meaningful keywords from text.
    
    Args:
        text: Input text
        min_length: Minimum word length to consider
    
    Returns:
        List of keywords
    """
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Split into words and filter
    words = text.split()
    keywords = [word for word in words if len(word) >= min_length]
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one',
        'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see',
        'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'very',
        'want', 'well', 'went', 'were', 'what', 'when', 'will', 'with', 'your', 'this', 'that',
        'they', 'have', 'from', 'each', 'which', 'their', 'time', 'would', 'there', 'could',
        'other', 'about', 'many', 'then', 'them', 'these', 'some', 'into', 'more', 'only',
        'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what', 'when', 'will',
        'with', 'your', 'been', 'call', 'come', 'does', 'down', 'even', 'find', 'first',
        'give', 'good', 'here', 'just', 'know', 'last', 'life', 'like', 'look', 'make',
        'most', 'much', 'must', 'name', 'need', 'next', 'part', 'people', 'same', 'seem',
        'should', 'tell', 'than', 'that', 'them', 'then', 'they', 'think', 'this', 'time',
        'very', 'want', 'well', 'went', 'were', 'what', 'when', 'will', 'with', 'work',
        'would', 'year', 'your', 'also', 'back', 'come', 'could', 'even', 'find', 'give',
        'here', 'just', 'know', 'last', 'life', 'like', 'look', 'make', 'most', 'much',
        'must', 'name', 'need', 'next', 'part', 'people', 'same', 'seem', 'should', 'tell',
        'than', 'that', 'them', 'then', 'they', 'think', 'this', 'time', 'very', 'want',
        'well', 'went', 'were', 'what', 'when', 'will', 'with', 'work', 'would', 'year'
    }
    
    keywords = [word for word in keywords if word not in stop_words]
    return keywords

def get_cluster_keywords(chapter_data: Dict[str, Dict], cluster_nodes: List[str]) -> List[str]:
    """
    Extract keywords from all chapters in a cluster.
    
    Args:
        chapter_data: Dictionary mapping chapter IDs to chapter info
        cluster_nodes: List of chapter IDs in the cluster
    
    Returns:
        List of keywords with frequencies
    """
    all_keywords = []
    
    for node_id in cluster_nodes:
        if node_id in chapter_data:
            chapter = chapter_data[node_id]
            
            # Extract keywords from chapter name
            if 'chapterName' in chapter:
                all_keywords.extend(extract_keywords(chapter['chapterName']))
            
            # Extract keywords from chapter summary
            if 'chapterSummary' in chapter:
                all_keywords.extend(extract_keywords(chapter['chapterSummary']))
    
    return all_keywords

def generate_cluster_label(keywords: List[str], max_words: int = 2) -> str:
    """
    Generate a meaningful label from keywords.
    
    Args:
        keywords: List of keywords with frequencies
        max_words: Maximum number of words in the label
    
    Returns:
        Generated label
    """
    if not keywords:
        return "General"
    
    # Count keyword frequencies
    keyword_counts = Counter(keywords)
    
    # Get top keywords
    top_keywords = [word for word, count in keyword_counts.most_common(max_words)]
    
    # Create label
    if len(top_keywords) == 1:
        return top_keywords[0].title()
    else:
        return " ".join(word.title() for word in top_keywords)

def label_hierarchy_tree(tree_node: Dict, chapter_data: Dict[str, Dict], level: int = 0) -> Dict:
    """
    Recursively label the hierarchy tree with meaningful names.
    
    Args:
        tree_node: Current node in the hierarchy tree
        chapter_data: Dictionary mapping chapter IDs to chapter info
        level: Current depth level
    
    Returns:
        Updated tree node with labels
    """
    # Generate label for current node
    if tree_node.get('type') == 'leaf' and 'nodes' in tree_node:
        # Leaf node - analyze chapter content
        keywords = get_cluster_keywords(chapter_data, tree_node['nodes'])
        label = generate_cluster_label(keywords)
        tree_node['label'] = label
    else:
        # Intermediate node - use children to generate label
        if 'children' in tree_node and tree_node['children']:
            all_keywords = []
            for child in tree_node['children']:
                if 'nodes' in child:
                    all_keywords.extend(get_cluster_keywords(chapter_data, child['nodes']))
            
            label = generate_cluster_label(all_keywords)
            tree_node['label'] = label
        else:
            tree_node['label'] = f"Level {level}"
    
    # Recursively process children
    if 'children' in tree_node:
        for child in tree_node['children']:
            label_hierarchy_tree(child, chapter_data, level + 1)
    
    return tree_node

def load_chapter_data_from_db() -> Dict[str, Dict]:
    """
    Load chapter data from database with full details.
    
    Returns:
        Dictionary mapping chapter IDs to chapter info
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from database import DatabaseManager, Chapter, Video
    
    db_manager = DatabaseManager()
    chapter_data = {}
    
    try:
        with db_manager.get_session() as session:
            # Get all chapters with video info
            chapters = session.query(
                Chapter.id,
                Chapter.chapter_name,
                Chapter.chapter_summary,
                Video.title.label('videoTitle')
            ).join(Video, Chapter.video_id == Video.id).all()
            
            for chapter in chapters:
                chapter_data[str(chapter.id)] = {
                    'id': str(chapter.id),
                    'chapterName': chapter.chapter_name,
                    'chapterSummary': chapter.chapter_summary,
                    'videoTitle': chapter.videoTitle
                }
    
    except Exception as e:
        print(f"Warning: Could not load chapter data from database: {e}")
        # Fallback to network graph data
        return load_chapter_data_from_network_graph()
    
    return chapter_data

def load_chapter_data_from_network_graph(network_graph_path: str = "../public/network-graph.json") -> Dict[str, Dict]:
    """
    Load chapter data from network graph JSON (fallback).
    
    Args:
        network_graph_path: Path to network-graph.json
    
    Returns:
        Dictionary mapping chapter IDs to chapter info
    """
    with open(network_graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chapter_data = {}
    for node in data['nodes']:
        node_id = node['data']['id']
        chapter_data[node_id] = {
            'id': node_id,
            'label': node['data']['label']
        }
    
    return chapter_data

def generate_labeled_hierarchy(hierarchy_path: str, output_path: str) -> Dict:
    """
    Generate hierarchy with meaningful labels.
    
    Args:
        hierarchy_path: Path to hierarchy.json
        output_path: Path to save labeled hierarchy
    
    Returns:
        Updated hierarchy data
    """
    print("Loading hierarchy data...")
    with open(hierarchy_path, 'r', encoding='utf-8') as f:
        hierarchy_data = json.load(f)
    
    print("Loading chapter data from database...")
    chapter_data = load_chapter_data_from_db()
    
    print("Generating labels...")
    labeled_tree = label_hierarchy_tree(hierarchy_data['tree'], chapter_data)
    hierarchy_data['tree'] = labeled_tree
    
    # Save labeled hierarchy
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hierarchy_data, f, indent=2, ensure_ascii=False)
    
    print(f"Labeled hierarchy saved to: {output_file}")
    return hierarchy_data

def main():
    """Generate labeled hierarchy."""
    hierarchy_path = "../public/hierarchy.json"
    output_path = "../public/labeled-hierarchy.json"
    
    generate_labeled_hierarchy(hierarchy_path, output_path)

if __name__ == "__main__":
    main() 