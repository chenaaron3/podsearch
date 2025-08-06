'use client';

import cytoscape from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import dagre from 'cytoscape-dagre';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '~/utils/api';

import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Separator } from './ui/separator';
import { Slider } from './ui/slider';

// Register Cytoscape extensions
cytoscape.use(dagre);
cytoscape.use(coseBilkent);

interface NetworkGraphProps {
    videoId?: number;
}

interface NodeData {
    id: string;
    label: string;
    videoId: number;
    videoTitle: string;
    chapterName: string;
    chapterSummary: string;
    startTime: number;
    endTime: number;
    duration: number;
}

interface EdgeData {
    id: string;
    source: string;
    target: string;
    similarityScore: number;
    weight: number;
}

interface NetworkData {
    nodes: { data: NodeData }[];
    edges: { data: EdgeData }[];
    stats: {
        totalNodes: number;
        totalEdges: number;
        averageSimilarity: number;
    };
}

export default function NetworkGraph({ videoId }: NetworkGraphProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const cyRef = useRef<cytoscape.Core | null>(null);
    const [similarityThreshold, setSimilarityThreshold] = useState(0.5);
    const [selectedNode, setSelectedNode] = useState<NodeData | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [layout, setLayout] = useState<'cose-bilkent' | 'dagre' | 'random'>('cose-bilkent');

    // Fetch network data
    const { data: networkData, refetch } = api.chapters.getNetworkData.useQuery(
        {
            similarityThreshold,
            limit: 1000,
            videoId,
        },
        {
            enabled: false, // We'll manually trigger this
        }
    );

    // Initialize Cytoscape
    const initializeCytoscape = useCallback((data: NetworkData) => {
        if (!containerRef.current) return;

        // Destroy existing instance
        if (cyRef.current) {
            cyRef.current.destroy();
        }

        // Create new instance
        cyRef.current = cytoscape({
            container: containerRef.current,
            elements: {
                nodes: data.nodes,
                edges: data.edges,
            },
            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': '#3b82f6',
                        'border-color': '#1e40af',
                        'border-width': 2,
                        'color': '#ffffff',
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'text-wrap': 'wrap',
                        'text-max-width': 120,
                        'font-size': 10,
                        'font-weight': 'bold',
                        'width': '60px',
                        'height': '60px',
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
                name: layout,
                animate: true,
                animationDuration: 1000,
                ...(layout === 'dagre' && {
                    rankDir: 'TB',
                    nodeDimensionsIncludeLabels: true,
                }),
                ...(layout === 'cose-bilkent' && {
                    nodeDimensionsIncludeLabels: true,
                    idealEdgeLength: 100,
                    nodeRepulsion: 4500,
                    gravity: 0.25,
                    gravityRange: 1.01,
                    gravityRangeCompound: 1.5,
                    gravityCompound: 1.0,
                    gravityNodeEdge: 0.25,
                    gravityEdge: 0.25,
                    gravityRangeEdge: 0.5,
                    gravityEdgeCompound: 1.0,
                    gravityRangeEdgeCompound: 0.5,
                    initialEnergyOnIncremental: 0.3,
                }),
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
            setSelectedNode(nodeData);
        });

        cyRef.current.on('tap', (evt) => {
            if (evt.target === cyRef.current) {
                setSelectedNode(null);
            }
        });

        // Fit to view
        cyRef.current.fit();
    }, [layout]);

    // Load data and initialize graph
    const loadGraph = useCallback(async () => {
        setIsLoading(true);
        try {
            const result = await refetch();
            if (result.data) {
                initializeCytoscape(result.data);
            }
        } catch (error) {
            console.error('Error loading graph data:', error);
        } finally {
            setIsLoading(false);
        }
    }, [refetch, initializeCytoscape]);

    // Initial load
    useEffect(() => {
        loadGraph();
    }, [loadGraph]);

    // Reload when threshold changes
    useEffect(() => {
        loadGraph();
    }, [similarityThreshold, loadGraph]);

    // Change layout
    const changeLayout = (newLayout: typeof layout) => {
        setLayout(newLayout);
        if (cyRef.current && networkData) {
            cyRef.current.layout({
                name: newLayout,
                animate: true,
                animationDuration: 1000,
                ...(newLayout === 'dagre' && {
                    rankDir: 'TB',
                    nodeDimensionsIncludeLabels: true,
                }),
                ...(newLayout === 'cose-bilkent' && {
                    nodeDimensionsIncludeLabels: true,
                    idealEdgeLength: 100,
                    nodeRepulsion: 4500,
                    gravity: 0.25,
                    gravityRange: 1.01,
                    gravityRangeCompound: 1.5,
                    gravityCompound: 1.0,
                    gravityNodeEdge: 0.25,
                    gravityEdge: 0.25,
                    gravityRangeEdge: 0.5,
                    gravityEdgeCompound: 1.0,
                    gravityRangeEdgeCompound: 0.5,
                    initialEnergyOnIncremental: 0.3,
                }),
            }).run();
        }
    };

    // Zoom controls
    const zoomIn = () => cyRef.current?.zoom({ level: cyRef.current.zoom() * 1.2, renderedPosition: { x: 0, y: 0 } });
    const zoomOut = () => cyRef.current?.zoom({ level: cyRef.current.zoom() * 0.8, renderedPosition: { x: 0, y: 0 } });
    const fitView = () => cyRef.current?.fit();

    // Format time
    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className="w-full h-full flex flex-col">
            {/* Controls */}
            <Card className="mb-4">
                <CardHeader>
                    <CardTitle className="text-lg">Network Controls</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="space-y-2">
                        <label className="text-sm font-medium">
                            Similarity Threshold: {similarityThreshold.toFixed(2)}
                        </label>
                        <Slider
                            value={[similarityThreshold]}
                            onValueChange={(value: number[]) => setSimilarityThreshold(value[0] || 0.5)}
                            min={0}
                            max={1}
                            step={0.01}
                            className="w-full"
                        />
                    </div>

                    <div className="flex gap-2">
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => changeLayout('cose-bilkent')}
                            disabled={layout === 'cose-bilkent'}
                        >
                            Force Layout
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => changeLayout('dagre')}
                            disabled={layout === 'dagre'}
                        >
                            Hierarchical
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => changeLayout('random')}
                            disabled={layout === 'random'}
                        >
                            Random
                        </Button>
                    </div>

                    <div className="flex gap-2">
                        <Button variant="outline" size="sm" onClick={zoomIn}>
                            Zoom In
                        </Button>
                        <Button variant="outline" size="sm" onClick={zoomOut}>
                            Zoom Out
                        </Button>
                        <Button variant="outline" size="sm" onClick={fitView}>
                            Fit View
                        </Button>
                        <Button variant="outline" size="sm" onClick={loadGraph} disabled={isLoading}>
                            {isLoading ? 'Loading...' : 'Refresh'}
                        </Button>
                    </div>

                    {networkData && (
                        <div className="flex gap-4 text-sm">
                            <Badge variant="secondary">
                                Nodes: {networkData.stats.totalNodes}
                            </Badge>
                            <Badge variant="secondary">
                                Edges: {networkData.stats.totalEdges}
                            </Badge>
                            <Badge variant="secondary">
                                Avg Similarity: {networkData.stats.averageSimilarity.toFixed(3)}
                            </Badge>
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* Graph Container */}
            <div className="flex-1 relative">
                <div
                    ref={containerRef}
                    className="w-full h-full border rounded-lg bg-white"
                    style={{ minHeight: '600px' }}
                />

                {/* Loading overlay */}
                {isLoading && (
                    <div className="absolute inset-0 bg-white/80 flex items-center justify-center">
                        <div className="text-lg">Loading network data...</div>
                    </div>
                )}
            </div>

            {/* Node Details Panel */}
            {selectedNode && (
                <Card className="mt-4">
                    <CardHeader>
                        <CardTitle className="text-lg">Chapter Details</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        <div>
                            <h3 className="font-semibold text-lg">{selectedNode.chapterName}</h3>
                            <p className="text-sm text-gray-600">{selectedNode.videoTitle}</p>
                        </div>

                        <Separator />

                        <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                                <span className="font-medium">Duration:</span>
                                <span className="ml-2">{formatTime(selectedNode.duration)}</span>
                            </div>
                            <div>
                                <span className="font-medium">Start Time:</span>
                                <span className="ml-2">{formatTime(selectedNode.startTime)}</span>
                            </div>
                        </div>

                        <div>
                            <span className="font-medium text-sm">Summary:</span>
                            <p className="text-sm text-gray-700 mt-1">{selectedNode.chapterSummary}</p>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
} 