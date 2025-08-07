'use client';

import cytoscape from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import dagre from 'cytoscape-dagre';
import { ChevronDown, ChevronRight, FileText, Folder, X } from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '~/utils/api';

import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

import type { NetworkGraphData, ChapterDetails } from '~/types/network-graph';

// Register Cytoscape extensions
cytoscape.use(dagre);
cytoscape.use(coseBilkent);

// Cluster colors for visualization
const clusterColors = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
    '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6366f1',
    '#14b8a6', '#f43f5e', '#a855f7', '#0ea5e9', '#22c55e',
    '#eab308', '#f97316', '#06b6d4', '#8b5cf6', '#ec4899'
];

interface NodeData {
    id: string;
    label: string;
}

interface HierarchyNode {
    id: string;
    name: string;
    label?: string;
    size: number;
    level: number;
    children?: HierarchyNode[];
    nodes?: string[];
    cluster_id?: number;
    type: 'leaf' | 'intermediate';
}

interface HierarchyData {
    tree: HierarchyNode;
    statistics: {
        total_clusters: number;
        total_nodes: number;
        largest_cluster: number;
        smallest_cluster: number;
        average_cluster_size: number;
    };
}

export default function NetworkGraph() {
    const containerRef = useRef<HTMLDivElement>(null);
    const cyRef = useRef<cytoscape.Core | null>(null);
    const [selectedNode, setSelectedNode] = useState<ChapterDetails | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [networkData, setNetworkData] = useState<NetworkGraphData | null>(null);
    const [hierarchyData, setHierarchyData] = useState<HierarchyData | null>(null);
    const [showHierarchy, setShowHierarchy] = useState(true);
    const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set(['root']));
    const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null);

    // Initialize Cytoscape
    const initializeCytoscape = useCallback((data: NetworkGraphData) => {
        if (!containerRef.current) return;

        // Destroy existing instance
        if (cyRef.current) {
            cyRef.current.destroy();
        }

        // Get visible elements based on hierarchy state
        const visibleNodes = getVisibleNodes();
        const visibleEdges = getVisibleEdges(visibleNodes);

        // Create new instance
        cyRef.current = cytoscape({
            container: containerRef.current,
            elements: {
                nodes: visibleNodes,
                edges: visibleEdges,
            },
            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': (ele: any) => {
                            const clusterId = ele.data('clusterId');
                            return clusterColors[clusterId % clusterColors.length] || '#3b82f6';
                        },
                        'border-color': '#1e40af',
                        'border-width': 2,
                        'color': '#ffffff',
                        'label': (ele: any) => {
                            const nodeData = ele.data();
                            if (nodeData.isBlob) {
                                return `${nodeData.label} (${nodeData.nodeCount})`;
                            }
                            return nodeData.label;
                        },
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'text-wrap': 'wrap',
                        'text-max-width': 120,
                        'font-size': 10,
                        'font-weight': 'bold',
                        'width': (ele: any) => {
                            const nodeData = ele.data();
                            if (nodeData.isBlob) {
                                // Proportional size based on node count, with min/max bounds
                                const baseSize = 60;
                                const sizeMultiplier = Math.min(Math.max(nodeData.nodeCount / 10, 1), 4);
                                return `${baseSize * sizeMultiplier}px`;
                            }
                            return '60px';
                        },
                        'height': (ele: any) => {
                            const nodeData = ele.data();
                            if (nodeData.isBlob) {
                                // Proportional size based on node count, with min/max bounds
                                const baseSize = 60;
                                const sizeMultiplier = Math.min(Math.max(nodeData.nodeCount / 10, 1), 4);
                                return `${baseSize * sizeMultiplier}px`;
                            }
                            return '60px';
                        },
                        'shape': 'ellipse',
                    },
                },
                {
                    selector: 'node:selected',
                    style: {
                        'background-color': '#ef4444',
                        'border-color': '#dc2626',
                        'border-width': 3,
                    },
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': '#94a3b8',
                        'curve-style': 'bezier',
                        'target-arrow-color': '#94a3b8',
                        'target-arrow-shape': 'triangle',
                        'arrow-scale': 0.5,
                        'opacity': 0.6,
                    },
                },
                {
                    selector: 'edge:selected',
                    style: {
                        'line-color': '#ef4444',
                        'target-arrow-color': '#ef4444',
                        'opacity': 1,
                        'width': 3,
                    },
                },
            ],
            layout: {
                name: 'cose-bilkent',
            },
            userZoomingEnabled: true,
            userPanningEnabled: true,
            boxSelectionEnabled: true,
            selectionType: 'single',
            touchTapThreshold: 8,
            desktopTapThreshold: 4,
            autoungrabify: false,
            autolock: false,
            autounselectify: false,
        });

        // Event listeners
        cyRef.current.on('tap', 'node', (evt) => {
            const node = evt.target;
            const nodeData = node.data() as NodeData;

            // Check if this is a blob node that can be expanded
            if (isNodeBlob(nodeData.id)) {
                // Extract hierarchy node ID from blob ID
                const hierarchyNodeId = nodeData.id.replace('blob_', '');
                toggleNodeExpansion(hierarchyNodeId);
            } else {
                // Normal node selection
                setSelectedChapterId(parseInt(nodeData.id));
            }
        });

        // Fit to view
        cyRef.current.fit();
    }, []);

    // State for fetching chapter details
    const [selectedChapterId, setSelectedChapterId] = useState<number | null>(null);

    // Fetch chapter details on demand
    const { data: chapterDetails, isLoading: isLoadingDetails } = api.chapters.getChapterDetails.useQuery(
        { chapterId: selectedChapterId! },
        { enabled: !!selectedChapterId }
    );

    // Update selected node when chapter details are loaded
    useEffect(() => {
        if (chapterDetails) {
            setSelectedNode(chapterDetails);
        }
    }, [chapterDetails]);

    // Load static network data
    const loadGraph = useCallback(async () => {
        setIsLoading(true);
        try {
            // Import the static JSON file
            const response = await fetch('/network-graph.json');
            if (!response.ok) {
                throw new Error(`Failed to load network graph: ${response.statusText}`);
            }
            const data: NetworkGraphData = await response.json();
            setNetworkData(data);
            initializeCytoscape(data);
        } catch (error) {
            console.error('Error loading graph data:', error);
        } finally {
            setIsLoading(false);
        }
    }, [initializeCytoscape]);

    // Initial load
    useEffect(() => {
        loadGraph();
    }, [loadGraph]);

    // Load hierarchy data
    useEffect(() => {
        const loadHierarchyData = async () => {
            try {
                // Try labeled hierarchy first, fallback to regular hierarchy
                let response = await fetch('/labeled-hierarchy.json');
                if (!response.ok) {
                    response = await fetch('/hierarchy.json');
                }

                if (response.ok) {
                    const data: HierarchyData = await response.json();
                    setHierarchyData(data);
                }
            } catch (error) {
                console.warn('Could not load hierarchy data:', error);
            }
        };

        loadHierarchyData();
    }, []);



    // Format time
    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    // Handle view video click
    const handleViewVideo = (chapterDetails: ChapterDetails) => {
        // Open video in new tab with timestamp
        const videoUrl = `https://www.youtube.com/watch?v=${chapterDetails.youtubeId}&t=${Math.floor(chapterDetails.startTime)}`;
        window.open(videoUrl, '_blank');
    };

    const handleClusterSelect = (clusterId: number) => {
        setSelectedClusterId(clusterId);
        // TODO: Highlight the selected cluster in the network graph
        console.log('Selected cluster:', clusterId);
    };

    const toggleNodeExpansion = (nodeId: string) => {
        setExpandedNodes(prev => {
            const newSet = new Set(prev);
            if (newSet.has(nodeId)) {
                newSet.delete(nodeId);
            } else {
                newSet.add(nodeId);
            }
            return newSet;
        });
    };

    const getVisibleNodes = useCallback(() => {
        if (!hierarchyData || !networkData) return networkData?.nodes || [];

        console.log('Hierarchy tree children count:', hierarchyData.tree.children?.length);
        console.log('Expanded nodes:', Array.from(expandedNodes));

        const visibleNodes: any[] = [];
        const blobNodes: any[] = [];

        const traverseHierarchy = (node: HierarchyNode) => {
            // Handle root node specially
            if (node.id === 'root') {
                if (node.children) {
                    node.children.forEach(child => traverseHierarchy(child));
                }
                return;
            }

            if (node.type === 'leaf') {
                // Leaf nodes - show individual chapters
                if (node.nodes) {
                    node.nodes.forEach(nodeId => {
                        const originalNode = networkData.nodes.find(n => n.data.id === nodeId);
                        if (originalNode) {
                            visibleNodes.push(originalNode);
                        }
                    });
                }
            } else if (node.children) {
                // Intermediate nodes
                if (expandedNodes.has(node.id)) {
                    // If expanded, show children
                    node.children.forEach(child => traverseHierarchy(child));
                } else {
                    // If collapsed, create a blob node proportional to size
                    // For intermediate nodes, we need to collect all leaf nodes under them
                    const collectLeafNodes = (hNode: HierarchyNode): string[] => {
                        if (hNode.type === 'leaf' && hNode.nodes) {
                            return hNode.nodes;
                        } else if (hNode.children) {
                            return hNode.children.flatMap(collectLeafNodes);
                        }
                        return [];
                    };

                    const leafNodes = collectLeafNodes(node);
                    if (leafNodes.length > 0) {
                        // Create a blob node representing the collapsed cluster
                        const blobNode = {
                            data: {
                                id: `blob_${node.id}`,
                                label: node.label || node.name,
                                clusterId: node.cluster_id || 0,
                                nodeCount: node.size,
                                isBlob: true,
                                originalNodes: leafNodes
                            }
                        };
                        blobNodes.push(blobNode);
                    }
                }
            }
        };

        traverseHierarchy(hierarchyData.tree);

        // Debug logging
        console.log('Visible nodes:', visibleNodes.length);
        console.log('Blob nodes:', blobNodes.length);
        console.log('Blob nodes:', blobNodes.map(b => ({ id: b.data.id, label: b.data.label, size: b.data.nodeCount })));

        // Return both individual nodes and blob nodes
        return [...visibleNodes, ...blobNodes];
    }, [hierarchyData, networkData, expandedNodes]);

    const getVisibleEdges = useCallback((visibleNodes: any[] = []) => {
        if (!hierarchyData || !networkData) return networkData?.edges || [];

        const visibleNodeIds = new Set<string>();
        const blobNodeIds = new Set<string>();

        const traverseHierarchy = (node: HierarchyNode) => {
            // Handle root node specially
            if (node.id === 'root') {
                if (node.children) {
                    node.children.forEach(child => traverseHierarchy(child));
                }
                return;
            }

            if (node.type === 'leaf') {
                if (node.nodes) {
                    node.nodes.forEach(nodeId => visibleNodeIds.add(nodeId));
                }
            } else if (node.children) {
                if (expandedNodes.has(node.id)) {
                    node.children.forEach(child => traverseHierarchy(child));
                } else {
                    // For intermediate nodes, collect all leaf nodes
                    const collectLeafNodes = (hNode: HierarchyNode): string[] => {
                        if (hNode.type === 'leaf' && hNode.nodes) {
                            return hNode.nodes;
                        } else if (hNode.children) {
                            return hNode.children.flatMap(collectLeafNodes);
                        }
                        return [];
                    };

                    const leafNodes = collectLeafNodes(node);
                    if (leafNodes.length > 0) {
                        // Add blob node ID
                        blobNodeIds.add(`blob_${node.id}`);
                        // Also add original nodes for edge calculations
                        leafNodes.forEach(nodeId => visibleNodeIds.add(nodeId));
                    }
                }
            }
        };

        traverseHierarchy(hierarchyData.tree);

        // Get the actual visible node IDs from the visibleNodes array
        const actualVisibleNodeIds = new Set(visibleNodes.map(node => node.data.id));

        // Only show edges between nodes that are actually in the visibleNodes array
        const visibleEdges = networkData.edges.filter(edge =>
            actualVisibleNodeIds.has(edge.data.source) && actualVisibleNodeIds.has(edge.data.target)
        );

        // For now, only show edges between visible individual nodes, not between blobs
        // This prevents the "nonexistent source" error
        return visibleEdges;
    }, [hierarchyData, networkData, expandedNodes]);

    const isNodeBlob = useCallback((nodeId: string): boolean => {
        return nodeId.startsWith('blob_');
    }, []);

    // Update graph when hierarchy state changes
    useEffect(() => {
        if (cyRef.current && networkData && hierarchyData) {
            const visibleNodes = getVisibleNodes();
            const visibleEdges = getVisibleEdges(visibleNodes);

            // Update graph elements
            cyRef.current.elements().remove();
            cyRef.current.add(visibleNodes);
            cyRef.current.add(visibleEdges);

            // Reapply layout
            cyRef.current.layout({ name: 'cose-bilkent' }).run();
            cyRef.current.fit();
        }
    }, [expandedNodes, networkData, hierarchyData, getVisibleNodes, getVisibleEdges]);

    // TreeNode component for hierarchy view
    const TreeNode: React.FC<{ node: HierarchyNode; level: number }> = ({ node, level }) => {
        const isExpanded = expandedNodes.has(node.id);
        const hasChildren = node.children && node.children.length > 0;
        const isLeaf = node.type === 'leaf';

        const handleToggle = () => {
            if (hasChildren) {
                toggleNodeExpansion(node.id);
            }
        };

        const handleClusterClick = () => {
            if (isLeaf && node.cluster_id !== undefined) {
                handleClusterSelect(node.cluster_id);
            }
        };

        const getNodeColor = (level: number) => {
            const colors = [
                'text-blue-600', 'text-green-600', 'text-purple-600',
                'text-orange-600', 'text-red-600', 'text-indigo-600'
            ];
            return colors[level % colors.length];
        };

        return (
            <div className="select-none">
                <div
                    className={`flex items-center py-1 px-2 hover:bg-gray-50 rounded cursor-pointer ${isLeaf ? 'hover:bg-blue-50' : ''
                        }`}
                    style={{ paddingLeft: `${level * 16 + 8}px` }}
                    onClick={isLeaf ? handleClusterClick : handleToggle}
                >
                    {hasChildren && (
                        <div className="mr-1">
                            {isExpanded ? (
                                <ChevronDown className="h-4 w-4 text-gray-500" />
                            ) : (
                                <ChevronRight className="h-4 w-4 text-gray-500" />
                            )}
                        </div>
                    )}

                    <div className="mr-2">
                        {isLeaf ? (
                            <FileText className="h-4 w-4 text-blue-500" />
                        ) : (
                            <Folder className="h-4 w-4 text-yellow-500" />
                        )}
                    </div>

                    <div className="flex-1">
                        <span
                            className={`font-medium ${getNodeColor(level)}`}
                            title={node.label && node.label !== node.name ? node.name : undefined}
                        >
                            {node.label || node.name}
                        </span>
                        <span className="text-sm text-gray-600 ml-2">
                            ({node.size} nodes)
                        </span>
                        {!isLeaf && !isExpanded && (
                            <span className="text-xs text-blue-500 ml-1">
                                🔵
                            </span>
                        )}
                    </div>
                </div>

                {hasChildren && isExpanded && (
                    <div>
                        {node.children!.map((child) => (
                            <TreeNode
                                key={child.id}
                                node={child}
                                level={level + 1}
                            />
                        ))}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="w-full h-full relative">
            {/* Graph Container */}
            <div
                ref={containerRef}
                className="w-full h-full bg-white"
            />

            {/* Loading overlay */}
            {isLoading && (
                <div className="absolute inset-0 bg-white/80 flex items-center justify-center">
                    <div className="text-center">
                        <div className="text-lg mb-2">Loading network graph...</div>
                        <div className="text-sm text-gray-600">Loading static network data</div>
                    </div>
                </div>
            )}

            {/* Cluster Overview Panel */}
            {networkData && (
                <div className="absolute top-4 left-4 bg-white/90 backdrop-blur-sm rounded-lg p-4 max-w-sm max-h-96 overflow-y-auto">
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold text-lg">Clusters ({networkData.clusters.length})</h3>
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setShowHierarchy(!showHierarchy)}
                            className="text-xs"
                        >
                            {showHierarchy ? 'List' : 'Tree'}
                        </Button>
                    </div>

                    {showHierarchy && hierarchyData ? (
                        <div className="space-y-2">
                            <TreeNode node={hierarchyData.tree} level={0} />
                        </div>
                    ) : (
                        <div className="space-y-2">
                            {networkData.clusters
                                .sort((a, b) => b.size - a.size) // Sort by size descending
                                .slice(0, 10)
                                .map((cluster) => (
                                    <div key={cluster.clusterId} className="flex items-center justify-between text-sm">
                                        <div className="flex items-center space-x-2">
                                            <div
                                                className="w-3 h-3 rounded-full"
                                                style={{
                                                    backgroundColor: clusterColors[cluster.clusterId % clusterColors.length] || '#3b82f6'
                                                }}
                                            />
                                            <span>Cluster {cluster.clusterId}</span>
                                        </div>
                                        <span className="text-gray-600">{cluster.size} nodes</span>
                                    </div>
                                ))}
                            {networkData.clusters.length > 10 && (
                                <div className="text-xs text-gray-500">
                                    +{networkData.clusters.length - 10} more clusters
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* Node Details Modal */}
            {(selectedNode || isLoadingDetails) && (
                <div className="absolute inset-0 bg-black/20 flex items-start justify-end p-4 pointer-events-none">
                    <Card className="w-96 max-h-[80vh] overflow-y-auto pointer-events-auto">
                        <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
                            <CardTitle className="text-lg">Chapter Details</CardTitle>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => {
                                    setSelectedNode(null);
                                    setSelectedChapterId(null);
                                }}
                                className="h-6 w-6 p-0"
                            >
                                <X className="h-4 w-4" />
                            </Button>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            {isLoadingDetails ? (
                                <div className="text-center py-8">
                                    <div className="text-sm text-gray-600">Loading chapter details...</div>
                                </div>
                            ) : selectedNode ? (
                                <>
                                    <div>
                                        <h3 className="font-semibold text-lg mb-1">{selectedNode.chapterName}</h3>
                                        <p className="text-sm text-gray-600">{selectedNode.videoTitle}</p>
                                    </div>

                                    <div className="space-y-2">
                                        <div className="flex justify-between text-sm">
                                            <span className="font-medium">Start Time:</span>
                                            <span>{formatTime(selectedNode.startTime)}</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="font-medium">Duration:</span>
                                            <span>{formatTime(selectedNode.duration)}</span>
                                        </div>
                                    </div>

                                    <div>
                                        <span className="font-medium text-sm">Summary:</span>
                                        <p className="text-sm text-gray-700 mt-1 leading-relaxed">
                                            {selectedNode.chapterSummary}
                                        </p>
                                    </div>

                                    <Button
                                        onClick={() => selectedNode && handleViewVideo(selectedNode)}
                                        className="w-full"
                                    >
                                        View Video
                                    </Button>
                                </>
                            ) : null}
                        </CardContent>
                    </Card>
                </div>
            )}
        </div>
    );
} 