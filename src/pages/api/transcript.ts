import type { NextApiRequest, NextApiResponse } from "next";
import { env } from '~/env';
import { db } from '~/server/db';
import { transcriptRequests } from '~/server/db/schema';

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
    console.log(
      `📹 Fetching transcript for YouTube ID: ${youtubeIdStr}, timestamp: ${timestampNum}s, duration: ${durationNum}s`,
    );

    // Fetch transcript from youtube-transcript.io API
    if (!env.YOUTUBE_TRANSCRIPT_API_KEY) {
      throw new Error("YouTube Transcript API key is not configured");
    }

    console.log(
      `🔑 Using API key: ${env.YOUTUBE_TRANSCRIPT_API_KEY ? "Present" : "Missing"}`,
    );

    const transcriptResponse = await fetch(
      "https://www.youtube-transcript.io/api/transcripts",
      {
        method: "POST",
        headers: {
          Authorization: `Basic ${env.YOUTUBE_TRANSCRIPT_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ids: [youtubeIdStr],
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

    // Find the first English track (prefer auto-generated)
    const englishTrack = tracks.find((track: TranscriptTrack) =>
      track.language?.toLowerCase().includes("en"),
    );

    if (!englishTrack?.transcript) {
      throw new Error("No English transcript available for this video");
    }

    const transcript: TranscriptSegment[] = englishTrack.transcript;
    console.log(`📄 Found English track: "${englishTrack.language}"`);
    console.log(`📄 Transcript segments: ${transcript.length}`);

    if (!transcript || transcript.length === 0) {
      throw new Error("No transcript content available for this video");
    }

    // Calculate the time range
    const startTimeSeconds = Math.max(0, timestampNum - durationNum);
    const endTimeSeconds = timestampNum;

    console.log(
      `⏰ Extracting transcript from ${startTimeSeconds}s to ${endTimeSeconds}s`,
    );

    // Filter transcript segments within the time range
    // The youtube-transcript.io API returns segments with start time and duration
    console.log(`⏰ Time range: ${startTimeSeconds}s to ${endTimeSeconds}s`);
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
      console.log(
        "⚠️ No transcript segments found in the specified time range",
      );
      transcriptText = "";
    } else {
      // Extract text from relevant segments
      console.log(
        `📄 Processing ${relevantSegments.length} relevant segments:`,
      );
      relevantSegments.forEach((segment: TranscriptSegment, index: number) => {
        console.log(`📄 Segment ${index + 1}: "${segment.text}"`);
      });

      transcriptText = relevantSegments
        .map((segment: TranscriptSegment) => segment.text)
        .join(" ")
        .trim();
    }

    console.log(
      `✅ Successfully extracted transcript: "${transcriptText.slice(0, 100)}..."`,
    );

    // Return the transcript
    res.status(200).json({
      transcript: transcriptText,
    });
  } catch (error) {
    success = false;
    errorMessage = error instanceof Error ? error.message : "Unknown error";
    console.error("❌ Transcript fetch failed:", error);

    // Return empty transcript on error
    res.status(200).json({
      transcript: "",
    });
  } finally {
    const processingTimeMs = Date.now() - startTime;

    // Log the request to database
    try {
      await db.insert(transcriptRequests).values({
        youtubeId: youtubeIdStr,
        timestamp: timestampNum,
        duration: durationNum,
        transcriptText: transcriptText,
        success,
        errorMessage,
        processingTimeMs,
      });

      console.log(
        `📊 Logged transcript request: ${youtubeIdStr} (${processingTimeMs}ms, success: ${success})`,
      );
    } catch (logError) {
      console.error("❌ Failed to log transcript request:", logError);
      // Don't throw - logging failure shouldn't break the API response
    }
  }
}
