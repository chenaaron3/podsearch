import { Filter, Search as SearchIcon } from 'lucide-react';
import Head from 'next/head';
import { useEffect, useRef, useState } from 'react';
import YouTube from 'react-youtube';
import { SearchBar } from '~/components/SearchBar';
import { SearchModal } from '~/components/SearchModal';
import { Button } from '~/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '~/components/ui/card';
import { Tabs, TabsList, TabsTrigger } from '~/components/ui/tabs';
import { useSearchStore } from '~/store/searchStore';
import { api } from '~/utils/api';

import type { YouTubePlayer, YouTubeRefs } from '~/types';

type SearchSegment = {
    id: string;
    score: number;
    youtubeId: string;
    videoTitle: string;
    segmentId: number;
    startTime: number;
    endTime: number;
    duration: number;
    timestampReadable: string;
    transcriptText: string;
    primaryEmotion?: string;
    emotionScore?: number;
};

type VideoGroup = {
    youtubeId: string;
    videoTitle: string;
    clips: SearchSegment[];
};

type SearchPhase = "input" | "results";

export default function Search() {
    const [currentPhase, setCurrentPhase] = useState<SearchPhase>("input");
    const [finalResults, setFinalResults] = useState<SearchSegment[]>([]);
    const [selectedClips, setSelectedClips] = useState<Record<string, number>>({}); // videoId -> clipIndex
    const [isSearching, setIsSearching] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState(0);
    const [searchQuery, setSearchQuery] = useState('');
    const youtubeRefs = useRef<YouTubeRefs>({});

    // Zustand store
    const { setLastGlobalSearch, lastGlobalQuery, lastGlobalResults } = useSearchStore();

    // Loading messages for animation
    const loadingMessages = [
        "Searching for relevant videos...",
        "Analyzing podcast segments...",
        "Finding key insights...",
        "Ranking best matches...",
        "Preparing your results..."
    ];
    const utils = api.useUtils();

    // Initialize from stored state if available
    useEffect(() => {
        if (lastGlobalResults.length > 0) {
            setFinalResults(lastGlobalResults);
            setCurrentPhase("results");
            setSearchQuery(lastGlobalQuery || '');
            initializeSelectedClips(lastGlobalResults);
        }
    }, [lastGlobalResults, lastGlobalQuery]);

    // Cycle through loading messages
    useEffect(() => {
        if (!isSearching) return;

        const interval = setInterval(() => {
            setLoadingMessage((prev) => {
                if (prev + 1 < loadingMessages.length) {
                    return prev + 1;
                } else {
                    return prev; // Stop at the last message
                }
            });
        }, 1500);

        return () => clearInterval(interval);
    }, [isSearching, loadingMessages.length]);

    // Group results by video
    const videoGroups: VideoGroup[] = finalResults.reduce((groups: VideoGroup[], segment) => {
        const existingGroup = groups.find(group => group.youtubeId === segment.youtubeId);

        if (existingGroup) {
            existingGroup.clips.push(segment);
        } else {
            groups.push({
                youtubeId: segment.youtubeId,
                videoTitle: segment.videoTitle,
                clips: [segment]
            });
        }

        return groups;
    }, []);

    // Initialize selected clips for new results
    const initializeSelectedClips = (results: SearchSegment[]) => {
        const groups = results.reduce((groups: VideoGroup[], segment) => {
            const existingGroup = groups.find(group => group.youtubeId === segment.youtubeId);

            if (existingGroup) {
                existingGroup.clips.push(segment);
            } else {
                groups.push({
                    youtubeId: segment.youtubeId,
                    videoTitle: segment.videoTitle,
                    clips: [segment]
                });
            }

            return groups;
        }, []);

        const newSelectedClips: Record<string, number> = {};
        groups.forEach(group => {
            if (!(group.youtubeId in selectedClips)) {
                // Default to first clip (most relevant due to ordering)
                newSelectedClips[group.youtubeId] = 0;
            }
        });
        setSelectedClips(prev => ({ ...prev, ...newSelectedClips }));
    };

    const handleSearch = async (query: string) => {
        try {
            setIsSearching(true);
            setLoadingMessage(0); // Reset to first message
            setSearchQuery(query);

            // Perform search
            const searchResults = await utils.search.search.fetch({
                query: query,
                topK: 5,
            });

            setFinalResults(searchResults.segments);
            setCurrentPhase("results");

            // Save to global state
            setLastGlobalSearch(query, searchResults.segments);

            // Initialize selected clips after setting results
            setTimeout(() => {
                initializeSelectedClips(searchResults.segments);
            }, 0);
        } catch (error) {
            console.error("Search failed:", error);
        } finally {
            setIsSearching(false);
        }
    };

    const getYouTubePlayerOptions = (videoGroup: VideoGroup) => ({
        height: '100%',
        width: '100%',
        playerVars: {
            start: Math.floor(videoGroup.clips[0]?.startTime ?? 0),
            autoplay: 0,
        },
    });

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const handleClipSelect = (videoId: string, clipIndex: number) => {
        setSelectedClips(prev => ({
            ...prev,
            [videoId]: clipIndex
        }));

        // Seek to the new timestamp if player is ready
        const player = youtubeRefs.current[videoId];
        if (player) {
            const videoGroup = videoGroups.find(group => group.youtubeId === videoId);
            if (videoGroup?.clips[clipIndex]) {
                const newStartTime = videoGroup.clips[clipIndex].startTime;
                player.seekTo(newStartTime, true);
                player.playVideo();
            }
        }
        // Pause all other players
        Object.keys(youtubeRefs.current).forEach(id => {
            if (id !== videoId) {
                const otherPlayer = youtubeRefs.current[id];
                if (otherPlayer) {
                    otherPlayer.pauseVideo();
                }
            }
        });
    };

    const createYouTubeReadyHandler = (videoId: string) => (event: { target: YouTubePlayer }) => {
        youtubeRefs.current[videoId] = event.target;
    };

    // Function to start searching within a specific video
    const startSearchWithinVideo = async (youtubeId: string) => {
        // Navigate to video search page
        window.location.href = `/search/${youtubeId}`;
    };

    // Touch/swipe functionality for mobile
    const [touchStart, setTouchStart] = useState<number | null>(null);
    const [touchEnd, setTouchEnd] = useState<number | null>(null);

    const minSwipeDistance = 50;

    const onTouchStart = (e: React.TouchEvent) => {
        setTouchEnd(null);
        setTouchStart(e.targetTouches[0]?.clientX ?? null);
    };

    const onTouchMove = (e: React.TouchEvent) => {
        setTouchEnd(e.targetTouches[0]?.clientX ?? null);
    };

    const onTouchEnd = (videoId: string, currentIndex: number, totalClips: number) => {
        if (!touchStart || !touchEnd) return;

        const distance = touchStart - touchEnd;
        const isLeftSwipe = distance > minSwipeDistance;
        const isRightSwipe = distance < -minSwipeDistance;

        if (isLeftSwipe && currentIndex < totalClips - 1) {
            // Swipe left - next clip
            handleClipSelect(videoId, currentIndex + 1);
        } else if (isRightSwipe && currentIndex > 0) {
            // Swipe right - previous clip
            handleClipSelect(videoId, currentIndex - 1);
        }
    };

    return (
        <>
            <Head>
                <title>Diary of a CEO Search</title>
                <meta name="description" content="Search through Diary of a CEO podcast episodes" />
                <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
                <link rel="icon" href="/favicon.ico" />
            </Head>

            <main className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
                <div className="container mx-auto px-4 sm:px-6 py-4 sm:py-8">
                    {/* Header */}
                    <div className="text-center mb-8 sm:mb-12">
                        <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-foreground mb-3 sm:mb-4 flex items-center justify-center gap-2 sm:gap-3">
                            <SearchIcon className="h-6 w-6 sm:h-8 sm:w-8 md:h-10 md:w-10 text-primary" />
                            <span className="leading-tight">Diary of a CEO Search</span>
                        </h1>
                        <p className="text-muted-foreground text-sm sm:text-base md:text-lg max-w-2xl mx-auto px-2">
                            Discover the most relevant insights from Steven Bartlett&apos;s podcast episodes with AI-powered search
                        </p>
                    </div>

                    {/* Persistent Search Bar - Show when in results phase */}
                    {currentPhase === "results" && (
                        <div className="max-w-2xl mx-auto mb-6 sm:mb-8 px-2">
                            <SearchBar
                                value={searchQuery}
                                onChange={setSearchQuery}
                                onSearch={() => searchQuery.trim() && handleSearch(searchQuery)}
                                placeholder="Search for topics, advice, insights..."
                                isSearching={isSearching}
                            />
                        </div>
                    )}

                    {/* Search Input Phase */}
                    {currentPhase === "input" && (
                        <SearchModal
                            title="What would you like to learn about?"
                            description="Search through thousands of podcast segments to find exactly what you need"
                            placeholder="e.g., building a successful startup, overcoming fear of failure, leadership advice, personal development..."
                            onSearch={handleSearch}
                            isSearching={isSearching}
                            loadingMessage={loadingMessages[loadingMessage]!}
                            buttonText="Search Episodes"
                        />
                    )}

                    {/* Results Phase */}
                    {currentPhase === "results" && (
                        <div className="space-y-6 sm:space-y-8">
                            {/* Video Groups */}
                            <div className="space-y-6 sm:space-y-8 max-w-7xl mx-auto">
                                {videoGroups.map((videoGroup) => {
                                    const selectedClipIndex = selectedClips[videoGroup.youtubeId] ?? 0;
                                    const selectedClip = videoGroup.clips[selectedClipIndex];

                                    // Safety check - if no clips or invalid index, skip rendering
                                    if (!selectedClip) {
                                        return null;
                                    }

                                    return (
                                        <Card key={videoGroup.youtubeId} className="shadow-lg border-0 bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm">
                                            <CardHeader className="pb-4 sm:pb-6">
                                                <div className="flex items-start justify-between">
                                                    <div className="flex-1">
                                                        <CardTitle className="text-lg sm:text-xl mb-2 line-clamp-2 leading-tight">
                                                            {videoGroup.videoTitle}
                                                        </CardTitle>
                                                    </div>
                                                </div>
                                            </CardHeader>
                                            <CardContent className="space-y-6 sm:space-y-8">
                                                {/* Video Player and Clip Content - Stacked on mobile, side-by-side on desktop */}
                                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8">
                                                    {/* Video Player */}
                                                    <div className="space-y-4 flex flex-col justify-center items-center">
                                                        <div
                                                            className="aspect-video rounded-lg overflow-hidden bg-black relative shadow-lg w-full max-w-full"
                                                            onTouchStart={onTouchStart}
                                                            onTouchMove={onTouchMove}
                                                            onTouchEnd={() => onTouchEnd(videoGroup.youtubeId, selectedClipIndex, videoGroup.clips.length)}
                                                        >
                                                            <YouTube
                                                                videoId={videoGroup.youtubeId}
                                                                opts={getYouTubePlayerOptions(videoGroup)}
                                                                className="w-full h-full"
                                                                key={videoGroup.youtubeId}
                                                                onReady={createYouTubeReadyHandler(videoGroup.youtubeId)}
                                                            />
                                                        </div>
                                                        <Button
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => {
                                                                void startSearchWithinVideo(videoGroup.youtubeId);
                                                            }}
                                                            className="text-xs text-muted-foreground hover:text-foreground"
                                                        >
                                                            <Filter className="h-3 w-3 mr-1" />
                                                            Search within this episode
                                                        </Button>
                                                    </div>

                                                    {/* Clip Navigation and Details */}
                                                    <div className="space-y-4 sm:space-y-6">
                                                        {/* Clip Tabs - Responsive grid */}
                                                        <div className="space-y-4">
                                                            <Tabs value={selectedClipIndex.toString()} onValueChange={(value) => handleClipSelect(videoGroup.youtubeId, parseInt(value))}>
                                                                <TabsList className="grid w-full grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 border h-auto">
                                                                    {videoGroup.clips.map((clip, index) => (
                                                                        <TabsTrigger
                                                                            key={clip.id}
                                                                            value={index.toString()}
                                                                            className="text-xs sm:text-sm py-2 sm:py-3 data-[state=inactive]:bg-muted/50 data-[state=inactive]:border data-[state=inactive]:border-border whitespace-nowrap"
                                                                        >
                                                                            <span className="hidden sm:inline">Clip {index + 1}</span>
                                                                            <span className="sm:hidden">C{index + 1}</span>
                                                                            <span className="ml-1 opacity-75">
                                                                                ({formatTime(clip.startTime)})
                                                                            </span>
                                                                        </TabsTrigger>
                                                                    ))}
                                                                </TabsList>
                                                            </Tabs>
                                                        </div>

                                                        {/* Selected Clip Details */}
                                                        <div className="space-y-4">
                                                            <div className="bg-muted/50 rounded-lg p-3 sm:p-4 text-sm leading-relaxed">
                                                                {selectedClip.transcriptText}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </>
    );
} 