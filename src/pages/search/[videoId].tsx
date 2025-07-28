import { ArrowLeft, Play, Search as SearchIcon, TrendingUp } from 'lucide-react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { useEffect, useState } from 'react';
import YouTube from 'react-youtube';
import { SearchBar } from '~/components/SearchBar';
import { SearchModal } from '~/components/SearchModal';
import { Timeline } from '~/components/Timeline';
import { Badge } from '~/components/ui/badge';
import { Button } from '~/components/ui/button';
import { Card, CardContent } from '~/components/ui/card';
import { Separator } from '~/components/ui/separator';
import { api } from '~/utils/api';

import type { SearchSegment } from "~/types";


type SearchPhase = "input" | "results";

export default function VideoSearch() {
    const router = useRouter();
    const { videoId } = router.query;

    const [currentPhase, setCurrentPhase] = useState<SearchPhase>("input");
    const [searchResults, setSearchResults] = useState<SearchSegment[]>([]);
    const [selectedClipIndex, setSelectedClipIndex] = useState(0);
    const [isSearching, setIsSearching] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState(0);
    const [searchQuery, setSearchQuery] = useState('');
    const [currentVideoTime, setCurrentVideoTime] = useState(0);

    // Loading messages for animation
    const loadingMessages = [
        "Searching within this episode...",
        "Analyzing video segments...",
        "Finding key insights...",
        "Ranking best matches...",
        "Preparing your results..."
    ];

    const utils = api.useUtils();

    // Get video details
    const { data: videoDetails, isLoading: isLoadingVideo } = api.search.getVideoByYoutubeId.useQuery(
        { youtubeId: videoId as string },
        { enabled: !!videoId }
    );

    // Cycle through loading messages
    useEffect(() => {
        if (!isSearching) return;

        const interval = setInterval(() => {
            setLoadingMessage((prev) => {
                if (prev + 1 < loadingMessages.length) {
                    return prev + 1;
                } else {
                    return prev;
                }
            });
        }, 1500);

        return () => clearInterval(interval);
    }, [isSearching, loadingMessages.length]);

    const handleSearch = async (query: string) => {
        if (!videoDetails?.id) return;

        try {
            setIsSearching(true);
            setLoadingMessage(0);
            setSearchQuery(query);

            const results = await utils.search.search.fetch({
                query,
                videoId: videoDetails.id,
                topK: 5,
            });

            setSearchResults(results.segments);
            setCurrentPhase("results");
            setSelectedClipIndex(0);
        } catch (error) {
            console.error("Search failed:", error);
        } finally {
            setIsSearching(false);
        }
    };

    const getYouTubePlayerOptions = (startTime: number) => ({
        height: '100%',
        width: '100%',
        playerVars: {
            start: Math.floor(startTime),
            autoplay: 0,
        },
    });

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const handleBackToGlobal = async () => {
        await router.push('/search');
    };

    const handleClipSelect = (clip: SearchSegment) => {
        const index = searchResults.findIndex(c => c.id === clip.id);
        setSelectedClipIndex(index);
        setCurrentVideoTime(clip.startTime);
    };

    const handleTimelineTimeUpdate = (time: number) => {
        setCurrentVideoTime(time);
    };

    if (isLoadingVideo) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
                    <p className="text-muted-foreground">Loading episode...</p>
                </div>
            </div>
        );
    }

    if (!videoDetails) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-2xl font-bold mb-4">Episode Not Found</h1>
                    <p className="text-muted-foreground mb-4">The episode you&apos;re looking for doesn&apos;t exist.</p>
                    <Button onClick={() => void handleBackToGlobal()}>
                        Back to Search
                    </Button>
                </div>
            </div>
        );
    }

    return (
        <>
            <Head>
                <title>Search - {videoDetails.title}</title>
                <meta name="description" content={`Search within ${videoDetails.title}`} />
                <link rel="icon" href="/favicon.ico" />
            </Head>

            <main className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
                <div className="container mx-auto px-4 py-8">
                    {/* Header */}
                    <div className="mb-8">
                        <div className="flex items-center justify-between mb-4">
                            <Button
                                variant="ghost"
                                onClick={() => void handleBackToGlobal()}
                                className="flex items-center gap-2"
                            >
                                <ArrowLeft className="h-4 w-4" />
                                Back to Global Search
                            </Button>
                        </div>
                    </div>

                    {/* Persistent Search Bar - Show when in results phase */}
                    {currentPhase === "results" && (
                        <div className="max-w-2xl mx-auto mb-8">
                            <SearchBar
                                value={searchQuery}
                                onChange={setSearchQuery}
                                onSearch={() => searchQuery.trim() && handleSearch(searchQuery)}
                                placeholder="Search within this episode..."
                                isSearching={isSearching}
                            />
                        </div>
                    )}

                    {/* Search Input Phase */}
                    {currentPhase === "input" && (
                        <SearchModal
                            title={`Search within: "${videoDetails.title ?? 'Episode'}"`}
                            description="Find specific moments and insights within this episode"
                            placeholder="e.g., startup advice, leadership tips, personal stories, business insights..."
                            onSearch={handleSearch}
                            isSearching={isSearching}
                            loadingMessage={loadingMessages[loadingMessage]!}
                            buttonText="Search Episode"
                            youtubeId={videoDetails.youtubeId}
                        />
                    )}

                    {/* Results Phase */}
                    {currentPhase === "results" && searchResults.length > 0 && (
                        <div className="space-y-6">
                            {/* Video Player and Clips Layout */}
                            <div className="grid lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
                                {/* Fixed Video Player */}
                                <div className="space-y-4">
                                    <div className="aspect-video rounded-lg overflow-hidden bg-black relative shadow-lg">
                                        <YouTube
                                            videoId={videoDetails.youtubeId}
                                            opts={getYouTubePlayerOptions(searchResults[selectedClipIndex]?.startTime ?? 0)}
                                            className="w-full h-full"
                                            key={`${videoDetails.youtubeId}-${searchResults[selectedClipIndex]?.startTime ?? 0}`}
                                        />
                                    </div>
                                </div>

                                {/* Selected Clip with Timeline */}
                                <div className="space-y-4">
                                    {/* Timeline */}
                                    <div className="space-y-2">
                                        <Timeline
                                            duration={videoDetails.duration ?? 3600} // Default to 1 hour if not available
                                            clips={searchResults}
                                            currentTime={currentVideoTime}
                                            onClipSelect={handleClipSelect}
                                            onTimeUpdate={handleTimelineTimeUpdate}
                                            className="w-full"
                                        />
                                    </div>

                                    {/* Selected Clip Details */}
                                    {searchResults[selectedClipIndex] && (
                                        <div className="space-y-4">
                                            <div className="flex items-center justify-between">
                                                <h3 className="text-lg font-semibold">Selected Clip</h3>
                                                <div className="flex items-center gap-2">
                                                    <span className="text-sm text-muted-foreground">
                                                        {selectedClipIndex + 1} of {searchResults.length}
                                                    </span>
                                                </div>
                                            </div>

                                            <Card className="ring-2 ring-primary">
                                                <CardContent className="p-6">
                                                    <div className="flex items-start justify-between mb-4">
                                                        <div className="flex items-center gap-3">
                                                            <Play className="h-5 w-5 text-primary" />
                                                            <div>
                                                                <span className="font-semibold text-lg">Clip {selectedClipIndex + 1}</span>
                                                                <span className="text-sm text-muted-foreground ml-2">
                                                                    ({formatTime(searchResults[selectedClipIndex].startTime)})
                                                                </span>
                                                            </div>
                                                        </div>
                                                        <Badge variant="default" className="gap-1">
                                                            <TrendingUp className="h-3 w-3" />
                                                            {Math.round((searchResults[selectedClipIndex].score ?? 0) * 100)}% match
                                                        </Badge>
                                                    </div>

                                                    <div className="space-y-3">
                                                        <Separator />

                                                        <div className="text-base leading-relaxed">
                                                            {searchResults[selectedClipIndex].transcriptText}
                                                        </div>
                                                    </div>
                                                </CardContent>
                                            </Card>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* No Results */}
                    {currentPhase === "results" && searchResults.length === 0 && (
                        <div className="text-center py-12">
                            <div className="max-w-md mx-auto">
                                <SearchIcon className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                                <h3 className="text-lg font-semibold mb-2">No results found</h3>
                                <p className="text-muted-foreground mb-4">
                                    Try adjusting your search terms or browse the full episode.
                                </p>
                                <Button onClick={() => setCurrentPhase("input")}>
                                    Try Another Search
                                </Button>
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </>
    );
} 