import console from 'console';
import { OpenAI } from 'openai';
import { env } from '~/env';
import { db } from '~/server/db';
import { transcriptRequests } from '~/server/db/schema';
import { GRAMMER_PROMPT, RANK_PROMPT, replacePromptPlaceholders } from '~/utils/llm';

import type { NextApiRequest, NextApiResponse } from "next";
const openai = new OpenAI({
  apiKey: env.OPENAI_API_KEY,
});

const OPENAI_MODEL = "gpt-4.1-nano";

// TypeScript interfaces for the YouTube Transcript API response
interface TranscriptSegment {
  text: string;
  start: string;
  dur: string;
}

interface TranscriptTrack {
  language: string;
  transcript: TranscriptSegment[];
}

interface VideoData {
  id: string;
  title: string;
  tracks: TranscriptTrack[];
  microformat?: {
    playerMicroformatRenderer?: {
      lengthSeconds?: string;
    };
  };
}

type TranscriptApiData = Array<VideoData>;

// API response types
interface TranscriptApiSuccessResponse {
  transcript: string;
}

interface TranscriptApiErrorResponse {
  error: string;
}

type TranscriptApiResponse =
  | TranscriptApiSuccessResponse
  | TranscriptApiErrorResponse;

interface GrammerOutput {
  transcript: string;
}

// Helper function 1: Get transcript from API
async function getTranscriptFromAPI(
  youtubeId: string,
): Promise<TranscriptSegment[]> {
  console.log(`📹 Fetching transcript for YouTube ID: ${youtubeId}`);

  if (!env.YOUTUBE_TRANSCRIPT_API_KEY) {
    throw new Error("YouTube Transcript API key is not configured");
  }

  console.log(
    `🔑 Using API key: ${env.YOUTUBE_TRANSCRIPT_API_KEY ? "Present" : "Missing"}`,
  );

  const keys = env.YOUTUBE_TRANSCRIPT_API_KEY.split(",");
  const key = keys[Math.floor(Math.random() * keys.length)];
  console.log(key);

  const transcriptResponse = await fetch(
    "https://www.youtube-transcript.io/api/transcripts",
    {
      method: "POST",
      headers: {
        Authorization: `Basic ${key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ids: [youtubeId],
      }),
    },
  );

  if (!transcriptResponse.ok) {
    if (transcriptResponse.status === 429) {
      throw new Error("Rate limit exceeded. Please try again later.");
    }
    throw new Error(`Transcript API error: ${transcriptResponse.status}`);
  }

  const rawData = (await transcriptResponse.json()) as TranscriptApiData;
  console.log("📄 Raw response type:", typeof rawData);
  console.log("📄 Raw response:", JSON.stringify(rawData, null, 2));

  // Type guard to ensure we have the expected structure
  if (!Array.isArray(rawData) || rawData.length === 0) {
    throw new Error("Invalid response format from transcript API");
  }

  const transcriptData = rawData;
  console.log(
    "📄 Complete Transcript API response:",
    JSON.stringify(transcriptData, null, 2),
  );
  console.log("📄 Response type:", typeof transcriptData);
  console.log("📄 Response keys:", Object.keys(transcriptData || {}));

  if (
    !transcriptData ||
    !Array.isArray(transcriptData) ||
    transcriptData.length === 0
  ) {
    throw new Error("No transcript available for this video");
  }

  const videoData = transcriptData[0]; // Get the first (and only) video
  if (!videoData) {
    throw new Error("No video data available");
  }

  console.log(`📄 Video ID in response: ${videoData.id}`);
  console.log(`📄 Video title: ${videoData.title}`);

  if (!videoData.tracks) {
    throw new Error("No transcript tracks available for this video");
  }

  // Extract transcript from the tracks array
  const tracks: TranscriptTrack[] = videoData.tracks || [];
  console.log(`📄 Number of tracks: ${tracks.length}`);

  // Prefer English; otherwise, just pick the first track
  const englishTrack = tracks.find((track: TranscriptTrack) =>
    track.language?.toLowerCase().includes("en"),
  );

  const selectedTrack: TranscriptTrack | undefined = englishTrack ?? tracks[0];

  if (!selectedTrack?.transcript || selectedTrack.transcript.length === 0) {
    throw new Error("No transcript content available for this video");
  }

  const transcript: TranscriptSegment[] = selectedTrack.transcript;
  const usedEnglish = Boolean(englishTrack && selectedTrack === englishTrack);
  console.log(
    `📄 Using ${usedEnglish ? "English" : "fallback"} track: "${selectedTrack.language}"`,
  );
  console.log(`📄 Transcript segments: ${transcript.length}`);

  if (!transcript || transcript.length === 0) {
    throw new Error("No transcript content available for this video");
  }

  return transcript;
}

// Helper function 2: Postprocess transcript with inputs
function postprocessTranscript(
  transcript: TranscriptSegment[],
  timestamp: number,
  duration: number,
): string {
  console.log(
    `⏰ Postprocessing transcript for timestamp: ${timestamp}s, duration: ${duration}s`,
  );

  // Calculate the time range
  const startTimeSeconds = Math.max(0, timestamp - duration);
  const endTimeSeconds = timestamp + 5;

  console.log(
    `⏰ Extracting transcript from ${startTimeSeconds}s to ${endTimeSeconds}s`,
  );
  console.log(`📄 Total transcript segments: ${transcript.length}`);

  const relevantSegments = transcript.filter((segment: TranscriptSegment) => {
    // Convert string timestamps to numbers
    const segmentStart = parseFloat(segment.start) || 0;
    const segmentDuration = parseFloat(segment.dur) || 0;
    const segmentEnd = segmentStart + segmentDuration;

    console.log(
      `📄 Segment: start=${segmentStart}s, end=${segmentEnd}s, text="${segment.text.slice(0, 50)}..."`,
    );

    // Check if segment overlaps with our time range
    const isRelevant =
      segmentStart <= endTimeSeconds && segmentEnd >= startTimeSeconds;
    if (isRelevant) {
      console.log(`✅ Segment is relevant!`);
    }
    return isRelevant;
  });

  console.log(`📄 Relevant segments found: ${relevantSegments.length}`);

  if (relevantSegments.length === 0) {
    console.log("⚠️ No transcript segments found in the specified time range");
    return "";
  }

  // Extract text from relevant segments
  console.log(`📄 Processing ${relevantSegments.length} relevant segments:`);
  relevantSegments.forEach((segment: TranscriptSegment, index: number) => {
    console.log(`📄 Segment ${index + 1}: "${segment.text}"`);
  });

  const transcriptText = relevantSegments
    .map((segment: TranscriptSegment) => segment.text)
    .join(" ")
    .trim();

  console.log(
    `✅ Successfully postprocessed transcript: "${transcriptText.slice(0, 100)}..."`,
  );

  return transcriptText;
}

// Helper function 3: Format transcript with GPT-4.1-nano
async function formatTranscriptWithGPT(
  transcriptText: string,
): Promise<string> {
  console.log(`🤖 Formatting transcript with GPT-4.1-nano`);

  const grammerPrompt = replacePromptPlaceholders(GRAMMER_PROMPT, {
    transcript: transcriptText,
  });

  const grammerCompletion = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    messages: [
      {
        role: "system",
        content: `You are an expert transcript cleaner.`,
      },
      {
        role: "user",
        content: grammerPrompt,
      },
    ],
    temperature: 0.3,
    max_tokens: 500,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "grammer_response",
        strict: true,
        schema: {
          type: "object",
          properties: {
            transcript: {
              type: "string",
              description: "The improved transcript with correct grammer",
            },
          },
          required: ["transcript"],
          additionalProperties: false,
        },
      },
    },
  });

  const grammerResponse = grammerCompletion.choices[0]?.message?.content;
  if (!grammerResponse) {
    throw new Error("No response from grammer model");
  }

  let enhancedTranscript = "";
  try {
    const grammerResults = JSON.parse(grammerResponse) as GrammerOutput;
    enhancedTranscript = grammerResults.transcript;
  } catch (error) {
    throw new Error("Invalid grammer response format");
  }

  console.log(
    `✅ Successfully formatted transcript with GPT: "${enhancedTranscript.slice(0, 100)}..."`,
  );

  return enhancedTranscript;
}

// Helper function 4: Store transcript in database
async function storeTranscriptInDB(
  youtubeId: string,
  timestamp: number,
  duration: number,
  transcriptText: string,
  success: boolean,
  errorMessage?: string,
  processingTimeMs?: number,
): Promise<void> {
  console.log(`💾 Storing transcript in database`);

  try {
    await db.insert(transcriptRequests).values({
      youtubeId,
      timestamp,
      duration,
      transcriptText,
      success,
      errorMessage,
      processingTimeMs,
    });

    console.log(
      `📊 Successfully logged transcript request: ${youtubeId} (${processingTimeMs}ms, success: ${success})`,
    );
  } catch (logError) {
    console.error("❌ Failed to log transcript request:", logError);
    // Don't throw - logging failure shouldn't break the API response
  }
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<TranscriptApiResponse>,
) {
  // Only allow GET requests
  if (req.method !== "GET") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { youtubeId, timestamp, duration } = req.query;

  // Validate required parameters
  if (!youtubeId || !timestamp || !duration) {
    return res.status(400).json({
      error: "Missing required parameters: youtubeId, timestamp, duration",
    });
  }

  // Validate parameter types
  const youtubeIdStr = String(youtubeId);
  const timestampNum = Number(timestamp);
  const durationNum = Number(duration);

  if (isNaN(timestampNum) || timestampNum < 0) {
    return res
      .status(400)
      .json({ error: "timestamp must be a non-negative number" });
  }

  if (isNaN(durationNum) || durationNum < 1) {
    return res
      .status(400)
      .json({ error: "duration must be a positive number" });
  }

  const startTime = Date.now();
  let success = true;
  let errorMessage: string | undefined;
  let transcriptText = "";

  try {
    // Step 1: Get transcript from API
    const transcript = await getTranscriptFromAPI(youtubeIdStr);

    // Step 2: Postprocess transcript with inputs
    transcriptText = postprocessTranscript(
      transcript,
      timestampNum,
      durationNum,
    );

    // Step 3: Format transcript with GPT-4.1-nano
    const enhancedTranscript = await formatTranscriptWithGPT(transcriptText);

    const processingTimeMs = Date.now() - startTime;

    // Step 4: Store in database
    await storeTranscriptInDB(
      youtubeIdStr,
      timestampNum,
      durationNum,
      enhancedTranscript,
      success,
      errorMessage,
      processingTimeMs,
    );

    // Return the enhanced transcript
    res.status(200).json({
      transcript: enhancedTranscript,
    });
  } catch (error) {
    success = false;
    errorMessage = error instanceof Error ? error.message : "Unknown error";
    console.error("❌ Transcript fetch failed:", error);

    const processingTimeMs = Date.now() - startTime;

    // Store error in database
    await storeTranscriptInDB(
      youtubeIdStr,
      timestampNum,
      durationNum,
      transcriptText,
      success,
      errorMessage,
      processingTimeMs,
    );

    // Return empty transcript on error
    res.status(200).json({
      transcript: "",
    });
  }
}
