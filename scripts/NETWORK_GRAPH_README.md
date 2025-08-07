# Network Graph Generation

This directory contains scripts to generate static network graph JSON files for the frontend.

## Overview

Instead of querying the database at runtime, we generate a static JSON file containing all network graph data. This approach:

- **Eliminates API bottlenecks**: No database queries during user interaction
- **Improves performance**: Static file loading is much faster
- **Reduces server load**: Database queries happen only during generation
- **Enables full dataset**: Can include all edges without pagination limits

## Files

- `generate_network_graph.py`: Main script to generate network graph JSON
- `generate_graph.sh`: Shell script wrapper for easy execution
- `database.py`: Database models and connection (shared with other scripts)

## Usage

### Generate Network Graph JSON

```bash
# Activate virtual environment
source venv/bin/activate

# Generate with default settings (threshold: 0.5)
python generate_network_graph.py

# Generate with custom threshold
python generate_network_graph.py --threshold 0.7

# Generate with custom output path
python generate_network_graph.py --output ../public/my-graph.json

# Generate with minimum component size filtering
python generate_network_graph.py --min-component-size 10

# Generate with high threshold and component filtering
python generate_network_graph.py --threshold 0.8 --min-component-size 4

# Use the shell script wrapper
./generate_graph.sh
```

### Parameters

- `--threshold`: Similarity threshold (0.0 to 1.0, default: 0.5)
- `--output`: Output JSON file path (default: ../public/network-graph.json)
- `--min-component-size`: Minimum nodes required in connected components (default: 4)

## Output Format

The generated JSON file contains minimal data for optimal performance:

```json
{
  "nodes": [
    {
      "data": {
        "id": "1",
        "label": "Chapter Name"
      }
    }
  ],
  "edges": [
    {
      "data": {
        "id": "1-2",
        "source": "1",
        "target": "2",
        "similarityScore": 0.85
      }
    }
  ],
  "stats": {
    "totalNodes": 9309,
    "totalEdges": 45491,
    "averageSimilarity": 0.627115
  },
  "metadata": {
    "generatedAt": "2024-08-06T22:02:00.000000",
    "similarityThreshold": 0.5,
    "version": "1.0"
  }
}
```

### Data Structure

- **Nodes**: Only contain `id` and `label` (chapter name)
- **Edges**: Only contain connection data and `similarityScore`
- **Chapter details**: Fetched on-demand via API when nodes are clicked

### Clustering

The script uses hierarchical clustering to create focused topic groups:

- **Hierarchical Clustering**: Recursively breaks down large clusters until all are under the size limit
- **Size Control**: All clusters are limited to a maximum size (default: 10 nodes)
- **Topic Focus**: Each cluster represents a very specific discussion area
- **Benefits**: Perfect for identifying and analyzing "hotspot" topics

### Component Filtering

The script automatically filters out clusters with fewer than the specified number of nodes:

- **Default**: 4+ nodes per cluster
- **Configurable**: Use `--min-component-size` parameter
- **Benefits**: Removes isolated nodes and small clusters for cleaner visualization

## Command Line Arguments

```bash
python generate_network_graph.py [OPTIONS]

Options:
  --threshold FLOAT        Similarity threshold (0.0 to 1.0, default: 0.5)
  --output PATH            Output JSON file path (default: ../public/network-graph.json)
  --min-component-size INT Minimum nodes per cluster (default: 4)
  --max-cluster-size INT   Maximum cluster size (default: 10)
  --help                   Show help message
```

## Frontend Integration

The frontend loads the JSON file from `/network-graph.json` and renders it using Cytoscape.js. The file is served as a static asset from the `public/` directory.

## Performance

- **File size**: ~8MB for 9,309 nodes and 45,491 edges (45% reduction)
- **Load time**: ~0.5-1 second on typical connections
- **Memory usage**: ~30-60MB in browser (reduced due to minimal data)
- **Rendering**: Cytoscape.js handles large graphs efficiently
- **On-demand loading**: Chapter details fetched only when needed

### Filtering Examples

- **Default (threshold 0.5, min 4 nodes)**: 9,260 nodes, 45,435 edges, 8MB
- **High threshold (0.8, min 4 nodes)**: 103 nodes, 238 edges, 48KB
- **Large components (threshold 0.5, min 10 nodes)**: 9,247 nodes, 45,396 edges, 8MB

## When to Regenerate

Regenerate the network graph JSON when:

- New videos/chapters are added to the database
- Similarity scores are recalculated
- You want to adjust the similarity threshold
- The graph structure changes significantly

## Automation

Consider adding this to your deployment pipeline:

```bash
# In your build script
cd scripts
source venv/bin/activate
python generate_network_graph.py
```

This ensures the network graph is always up-to-date with the latest data.
