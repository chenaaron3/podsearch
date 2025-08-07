export interface NetworkGraphNode {
  data: {
    id: string;
    label: string;
    clusterId: number;
  };
}

export interface NetworkGraphEdge {
  data: {
    id: string;
    source: string;
    target: string;
    similarityScore: number;
  };
}

export interface NetworkGraphStats {
  totalNodes: number;
  totalEdges: number;
  averageSimilarity: number;
}

export interface NetworkGraphMetadata {
  generatedAt: string;
  similarityThreshold: number;
  version: string;
}

export interface NetworkGraphCluster {
  clusterId: number;
  size: number;
  nodeIds: string[];
}

export interface NetworkGraphData {
  nodes: NetworkGraphNode[];
  edges: NetworkGraphEdge[];
  clusters: NetworkGraphCluster[];
  stats: NetworkGraphStats;
  metadata: NetworkGraphMetadata;
}

// Type for detailed node information fetched on demand
export interface ChapterDetails {
  id: number;
  videoId: number;
  youtubeId: string;
  videoTitle: string;
  chapterName: string;
  chapterSummary: string;
  startTime: number;
  endTime: number;
  duration: number;
}
