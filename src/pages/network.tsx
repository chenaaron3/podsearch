import type { NextPage } from 'next';
import Head from 'next/head';
import { useState } from 'react';
import NetworkGraph from '~/components/NetworkGraph';
import { Badge } from '~/components/ui/badge';
import { Button } from '~/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '~/components/ui/card';
import {
    Select, SelectContent, SelectItem, SelectTrigger, SelectValue
} from '~/components/ui/select';
import { api } from '~/utils/api';

const NetworkPage: NextPage = () => {
    const [selectedVideoId, setSelectedVideoId] = useState<number | undefined>(undefined);

    // Get all videos for the dropdown
    const { data: videos } = api.chapters.getStats.useQuery();

    // Get video list for dropdown
    const { data: videoList } = api.videos.getWithChapters.useQuery(
        { limit: 1000 },
        { enabled: true }
    );

    return (
        <>
            <Head>
                <title>Chapter Network Visualization - PodSearch</title>
                <meta name="description" content="Interactive network visualization of chapter similarities" />
            </Head>

            <main className="container mx-auto px-4 py-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold mb-2">Chapter Network Visualization</h1>
                    <p className="text-gray-600">
                        Explore the relationships between podcast chapters based on similarity scores.
                        Adjust the similarity threshold to see different clusters and connections.
                    </p>
                </div>

                {/* Stats Overview */}
                {videos && (
                    <Card className="mb-6">
                        <CardHeader>
                            <CardTitle>Database Overview</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="flex gap-4">
                                <Badge variant="outline">
                                    Total Videos: {videos.totalVideos}
                                </Badge>
                                <Badge variant="outline">
                                    Total Chapters: {videos.totalChapters}
                                </Badge>
                                <Badge variant="outline">
                                    Total Similarities: {videos.totalSimilarities}
                                </Badge>
                            </div>
                        </CardContent>
                    </Card>
                )}

                {/* Video Filter */}
                <Card className="mb-6">
                    <CardHeader>
                        <CardTitle>Filter by Video</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="flex gap-4 items-center">
                            <Select
                                value={selectedVideoId?.toString() || "all"}
                                onValueChange={(value) => setSelectedVideoId(value === "all" ? undefined : parseInt(value))}
                            >
                                <SelectTrigger className="w-[300px]">
                                    <SelectValue placeholder="Select a video to filter" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="all">All Videos</SelectItem>
                                    {videoList?.map((video) => (
                                        <SelectItem key={video.id} value={video.id.toString()}>
                                            {video.title}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>

                            {selectedVideoId && (
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => setSelectedVideoId(undefined)}
                                >
                                    Clear Filter
                                </Button>
                            )}
                        </div>
                    </CardContent>
                </Card>

                {/* Network Graph */}
                <div className="h-[800px]">
                    <NetworkGraph videoId={selectedVideoId} />
                </div>

                {/* Instructions */}
                <Card className="mt-8">
                    <CardHeader>
                        <CardTitle>How to Use</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <h4 className="font-semibold mb-2">Controls</h4>
                                <ul className="text-sm space-y-1 text-gray-600">
                                    <li>• <strong>Similarity Threshold:</strong> Adjust to show only connections above a certain similarity score</li>
                                    <li>• <strong>Layout:</strong> Choose between force-directed, hierarchical, or random layouts</li>
                                    <li>• <strong>Zoom:</strong> Use zoom controls or mouse wheel to explore the network</li>
                                    <li>• <strong>Pan:</strong> Click and drag to move around the visualization</li>
                                </ul>
                            </div>
                            <div>
                                <h4 className="font-semibold mb-2">Interaction</h4>
                                <ul className="text-sm space-y-1 text-gray-600">
                                    <li>• <strong>Click nodes:</strong> View detailed chapter information</li>
                                    <li>• <strong>Node size:</strong> Larger nodes may indicate more connections</li>
                                    <li>• <strong>Edge thickness:</strong> Thicker edges indicate higher similarity scores</li>
                                    <li>• <strong>Clusters:</strong> Similar chapters will cluster together</li>
                                </ul>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </main>
        </>
    );
};

export default NetworkPage; 