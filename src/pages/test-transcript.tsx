import { useState } from 'react';

// TypeScript interfaces for the API response
interface TranscriptApiSuccessResponse {
    transcript: string;
}

interface TranscriptApiErrorResponse {
    error: string;
}

type TranscriptApiResponse = TranscriptApiSuccessResponse | TranscriptApiErrorResponse;

// TypeScript interfaces for form data
interface FormData {
    youtubeId: string;
    timestamp: string;
    duration: string;
}

// TypeScript interface for API request parameters
interface ApiRequestParams {
    youtubeId: string;
    timestamp: number;
    duration: number;
}

export default function TestTranscript() {
    const [youtubeId, setYoutubeId] = useState<string>('dQw4w9WgXcQ');
    const [timestamp, setTimestamp] = useState<string>('60');
    const [duration, setDuration] = useState<string>('30');
    const [result, setResult] = useState<string>('');
    const [loading, setLoading] = useState<boolean>(false);

    const testAPI = async (): Promise<void> => {
        setLoading(true);
        try {
            // Validate input parameters
            const timestampNum: number = parseInt(timestamp, 10);
            const durationNum: number = parseInt(duration, 10);

            if (isNaN(timestampNum) || timestampNum < 0) {
                throw new Error('Timestamp must be a non-negative number');
            }

            if (isNaN(durationNum) || durationNum < 1) {
                throw new Error('Duration must be a positive number');
            }

            if (!youtubeId.trim()) {
                throw new Error('YouTube ID is required');
            }

            const params: ApiRequestParams = {
                youtubeId: youtubeId.trim(),
                timestamp: timestampNum,
                duration: durationNum
            };

            const response: Response = await fetch(
                `/api/transcript?youtubeId=${params.youtubeId}&timestamp=${params.timestamp}&duration=${params.duration}`
            );

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data: TranscriptApiResponse = await response.json() as TranscriptApiResponse;
            setResult(JSON.stringify(data, null, 2));
        } catch (error) {
            const errorMessage: string = error instanceof Error ? error.message : 'Unknown error';
            setResult(`Error: ${errorMessage}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container mx-auto p-8 max-w-4xl">
            <h1 className="text-3xl font-bold mb-6">Transcript API Test</h1>

            <div className="space-y-4 mb-6">
                <div>
                    <label className="block text-sm font-medium mb-2">YouTube ID:</label>
                    <input
                        type="text"
                        value={youtubeId}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setYoutubeId(e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded"
                        placeholder="e.g., dQw4w9WgXcQ"
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium mb-2">Timestamp (seconds):</label>
                    <input
                        type="number"
                        value={timestamp}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTimestamp(e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded"
                        min="0"
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium mb-2">Duration (seconds):</label>
                    <input
                        type="number"
                        value={duration}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setDuration(e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded"
                        min="1"
                    />
                </div>

                <button
                    onClick={testAPI}
                    disabled={loading}
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                    type="button"
                >
                    {loading ? 'Testing...' : 'Test API'}
                </button>
            </div>

            {result && (
                <div>
                    <h2 className="text-xl font-semibold mb-2">Result:</h2>
                    <pre className="bg-gray-100 p-4 rounded overflow-auto max-h-96">
                        {result}
                    </pre>
                </div>
            )}

            <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded">
                <h3 className="font-semibold mb-2">API Endpoint:</h3>
                <code className="text-sm">
                    GET /api/transcript?youtubeId=YOUR_YOUTUBE_ID&timestamp=SECONDS&duration=SECONDS
                </code>
                <p className="text-sm mt-2 text-gray-600">
                    Returns: {"{ transcript: string }"}
                </p>
            </div>
        </div>
    );
} 