import Head from 'next/head';
import { useState } from 'react';
import YouTube from 'react-youtube';
import { api } from '~/utils/api';

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

type SearchPhase = "input" | "results";

export default function Search() {
    const [searchQuery, setSearchQuery] = useState("");
    const [currentPhase, setCurrentPhase] = useState<SearchPhase>("input");
    const [finalResults, setFinalResults] = useState<SearchSegment[]>([]);

    // tRPC queries and mutations
    const [isSearching, setIsSearching] = useState(false);
    const utils = api.useUtils();

    const handleSearch = async () => {
        if (!searchQuery.trim()) return;

        try {
            setIsSearching(true);

            // Perform smart search
            const searchResults = await utils.search.smartSearch.fetch({
                query: searchQuery,
                topK: 5,
            });

            setFinalResults(searchResults.segments);
            setCurrentPhase("results");
        } catch (error) {
            console.error("Search failed:", error);
        } finally {
            setIsSearching(false);
        }
    };

    const getYouTubePlayerOptions = (startTime: number) => ({
        height: '315',
        width: '560',
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

    const resetSearch = () => {
        setSearchQuery("");
        setCurrentPhase("input");
        setFinalResults([]);
    };

    return (
        <>
            <Head>
                <title>Diary of a CEO Search</title>
                <meta name="description" content="Search through Diary of a CEO podcast episodes" />
                <link rel="icon" href="/favicon.ico" />
            </Head>

            <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
                <div className="container mx-auto px-4 py-8">
                    {/* Header */}
                    <div className="text-center mb-8">
                        <h1 className="text-4xl font-bold text-white mb-4">
                            Diary of a CEO Search
                        </h1>
                        <p className="text-slate-300 text-lg">
                            Find the perfect clips from Steven Bartlett&apos;s podcast
                        </p>
                    </div>

                    {/* Search Input Phase */}
                    {currentPhase === "input" && (
                        <div className="max-w-2xl mx-auto">
                            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20">
                                <h2 className="text-2xl font-semibold text-white mb-6 text-center">
                                    What are you interested in learning about?
                                </h2>

                                <div className="space-y-4">
                                    <textarea
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                        placeholder="e.g., building a successful startup, overcoming fear of failure, leadership advice..."
                                        className="w-full p-4 rounded-xl bg-white/10 border border-white/30 text-white placeholder-slate-400 resize-none"
                                        rows={3}
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter' && !e.shiftKey) {
                                                e.preventDefault();
                                                void handleSearch();
                                            }
                                        }}
                                    />

                                    <button
                                        onClick={() => void handleSearch()}
                                        disabled={!searchQuery.trim() || isSearching}
                                        className="w-full py-3 px-6 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-xl hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                                    >
                                        {isSearching ? "Finding your clips..." : "Search Episodes"}
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Results Phase */}
                    {currentPhase === "results" && (
                        <div className="space-y-8">
                            <div className="text-center">
                                <h2 className="text-3xl font-bold text-white mb-2">
                                    Your Personalized Clips
                                </h2>
                                <p className="text-slate-300 mb-4">
                                    Found {finalResults.length} clips based on your query
                                </p>
                                <button
                                    onClick={resetSearch}
                                    className="px-6 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-all"
                                >
                                    New Search
                                </button>
                            </div>

                            {/* Results Grid */}
                            <div className="grid gap-8 max-w-6xl mx-auto">
                                {finalResults.map((segment) => (
                                    <div key={segment.id} className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
                                        <div className="grid md:grid-cols-2 gap-6">
                                            {/* Video Player */}
                                            <div className="space-y-4">
                                                <div className="aspect-video rounded-xl overflow-hidden bg-black">
                                                    <YouTube
                                                        videoId={segment.youtubeId}
                                                        opts={getYouTubePlayerOptions(segment.startTime)}
                                                        className="w-full h-full"
                                                    />
                                                </div>

                                                <div className="flex items-center justify-between text-sm text-slate-300">
                                                    <span>
                                                        Starts at {formatTime(segment.startTime)} • {Math.round(segment.duration)}s clip
                                                    </span>
                                                    <span className="bg-purple-600/30 px-2 py-1 rounded">
                                                        Match: {Math.round(segment.score * 100)}%
                                                    </span>
                                                </div>
                                            </div>

                                            {/* Transcript and Details */}
                                            <div className="space-y-4">
                                                <div>
                                                    <h3 className="text-xl font-semibold text-white mb-2 line-clamp-2">
                                                        {segment.videoTitle}
                                                    </h3>
                                                    <p className="text-sm text-slate-400 mb-3">
                                                        Segment {segment.segmentId} • {segment.timestampReadable}
                                                        {segment.primaryEmotion && (
                                                            <span className="ml-2 bg-blue-600/30 px-2 py-0.5 rounded text-xs">
                                                                {segment.primaryEmotion}
                                                            </span>
                                                        )}
                                                    </p>
                                                </div>

                                                <div className="bg-white/5 rounded-xl p-4">
                                                    <h4 className="text-sm font-medium text-slate-300 mb-2">Transcript</h4>
                                                    <p className="text-white text-sm leading-relaxed">
                                                        {segment.transcriptText}
                                                    </p>
                                                </div>

                                                <a
                                                    href={`https://youtube.com/watch?v=${segment.youtubeId}&t=${Math.floor(segment.startTime)}s`}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="inline-flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-all text-sm"
                                                >
                                                    Watch on YouTube
                                                    <svg className="w-4 h-4 ml-2" fill="currentColor" viewBox="0 0 20 20">
                                                        <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                                                    </svg>
                                                </a>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </>
    );
} 