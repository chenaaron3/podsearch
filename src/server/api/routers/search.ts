import { eq, inArray } from 'drizzle-orm';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import OpenAI from 'openai';
import { join } from 'path';
import { z } from 'zod';
import { env } from '~/env';
import { createTRPCRouter, publicProcedure } from '~/server/api/trpc';
import { searchExecutions, transcripts, videos } from '~/server/db/schema';
import { RANK_PROMPT, replacePromptPlaceholders, SEEK_PROMPT } from '~/utils/llm';

import { Pinecone } from '@pinecone-database/pinecone';

import type {
  TranscriptData,
  TranscriptSegment,
  WordTimestamp,
  PineconeSearchQuery,
  SearchResponse,
  TRPCContext,
  InputClipMetadata,
  OutputSegmentMetadata,
} from "~/types";
// Initialize clients
const pinecone = new Pinecone({
  apiKey: env.PINECONE_API_KEY,
});

const openai = new OpenAI({
  apiKey: env.OPENAI_API_KEY,
});

const OPENAI_REASONING_MODEL = "gpt-4o-mini";
const OPENAI_MODEL = "gpt-4.1-mini";
const PINECONE_INDEX_NAME = "video-segments";
// Types for Pinecone metadata
interface PineconeMetadata {
  video_id: number;
  video_name: string;
  segment_id: number;
  start_time: number;
  end_time: number;
  duration: number;
  timestamp_readable: string;
  primary_emotion?: string;
  primary_emotion_score?: number;
}

// Types for structured LLM inputs and outputs
interface RankingOutputItem {
  clipIndex: number;
  reasoning: string;
}

type RankingOutput = RankingOutputItem[];

interface SeekingWordInputItem {
  w: string;
  i: number;
}

type SeekingWordInput = SeekingWordInputItem[];

interface SeekingOutput {
  start_index: number;
  reasoning: string;
}

// Types for clip processing
interface ClipForRanking {
  clip_id: string;
  video_id: number;
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
  similarityScore: number;
  transcriptData: TranscriptData;
}

// Validation schemas for LLM outputs
const RankingOutputSchema = z.array(
  z.object({
    clipIndex: z.number().int().min(0),
    reasoning: z.string().min(1),
  }),
);

const SeekingOutputSchema = z.object({
  start_index: z.number().int().min(0),
  reasoning: z.string().min(1),
});

// Types for search results
const SearchSegment = z.object({
  id: z.string(),
  score: z.number(),
  youtubeId: z.string(),
  videoTitle: z.string(),
  segmentId: z.number(),
  startTime: z.number(),
  endTime: z.number(),
  duration: z.number(),
  timestampReadable: z.string(),
  transcriptText: z.string(),
  primaryEmotion: z.string().optional(),
  emotionScore: z.number().optional(),
});

// Helper function to extract word-level timestamps from transcript segments within a time range
function extractWordTimestamps(
  transcriptSegments: TranscriptSegment[],
  startTime: number,
  endTime: number,
): WordTimestamp[] {
  const words: WordTimestamp[] = [];

  for (const segment of transcriptSegments) {
    const segStart = segment.start || 0;
    const segEnd = segment.end || segStart + 1;

    // Only process segments that overlap with the given time range
    const tolerance = 1.0;
    if (segStart - tolerance <= endTime && segEnd + tolerance >= startTime) {
      if (segment.words && Array.isArray(segment.words)) {
        for (const word of segment.words) {
          if (
            word.word &&
            typeof word.start === "number" &&
            typeof word.end === "number"
          ) {
            words.push({
              word: word.word,
              start: word.start,
              end: word.end,
            });
          }
        }
      }
    }
  }

  return words;
}

// Helper function to get clip transcript segments from video transcript using timestamp ranges
function getClipTranscriptSegmentsByTimestamp(
  transcriptData: TranscriptData | { segments: unknown },
  startTime: number,
  endTime: number,
): TranscriptSegment[] {
  if (!transcriptData?.segments) return [];

  const segments = Array.isArray(transcriptData.segments)
    ? (transcriptData.segments as TranscriptSegment[])
    : (JSON.parse(transcriptData.segments as string) as TranscriptSegment[]);

  // Find transcript segments that overlap with the given time range
  const overlappingSegments = segments.filter((seg: TranscriptSegment) => {
    const segStart = seg.start || 0;
    const segEnd = seg.end || segStart + 1;

    // Check for overlap with some tolerance (±1 second)
    const tolerance = 1.0;
    return segStart - tolerance <= endTime && segEnd + tolerance >= startTime;
  });

  // Sort by start time
  overlappingSegments.sort(
    (a: TranscriptSegment, b: TranscriptSegment) =>
      (a.start || 0) - (b.start || 0),
  );
  return overlappingSegments;
}

// Helper function to join transcript segments into text
function joinTranscriptSegments(segments: TranscriptSegment[]): string {
  return segments
    .map((seg: TranscriptSegment) => seg.text || "")
    .join(" ")
    .trim();
}

// Helper function to get clip transcript from video transcript using timestamp ranges
function getClipTranscriptByTimestamp(
  transcriptData: TranscriptData | { segments: unknown },
  startTime: number,
  endTime: number,
): string {
  const segments = getClipTranscriptSegmentsByTimestamp(
    transcriptData,
    startTime,
    endTime,
  );
  return joinTranscriptSegments(segments);
}

// Helper function to save prompts and responses to filesystem
function savePromptResponse(
  step: string,
  prompt: string,
  response: string,
  metadata?: Record<string, unknown>,
): void {
  try {
    const logsDir = join(process.cwd(), "logs");
    if (!existsSync(logsDir)) {
      mkdirSync(logsDir, { recursive: true });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `${step}_${timestamp}.json`;
    const filepath = join(logsDir, filename);

    const logData = {
      timestamp: new Date().toISOString(),
      step,
      prompt,
      response,
      metadata,
    };

    writeFileSync(filepath, JSON.stringify(logData, null, 2));
    console.log(`📝 Saved prompt/response to: ${filepath}`);
  } catch (error) {
    console.error("❌ Error saving prompt/response:", error);
  }
}

// Helper function to log search executions to database
async function logSearchExecution(
  ctx: TRPCContext,
  query: string,
  videoId: number | undefined,
  topK: number,
  inputClipsCount: number,
  outputSegmentsCount: number,
  inputClipsMetadata: InputClipMetadata[],
  outputSegmentsMetadata: OutputSegmentMetadata[],
  processingTimeMs: number,
  status: "success" | "error",
  errorMessage?: string,
): Promise<void> {
  try {
    // Get user info from session if available
    const userId = ctx.session?.user?.id;

    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    await ctx.db.insert(searchExecutions).values({
      userId,
      query,
      videoId,
      topK,
      inputClipsCount,
      outputSegmentsCount,
      inputClipsMetadata,
      outputSegmentsMetadata,
      processingTimeMs,
      status,
      errorMessage,
    });

    console.log(
      `📊 Logged search execution: "${query}" (${processingTimeMs}ms, ${outputSegmentsCount} results, status: ${status})`,
    );
  } catch (error) {
    console.error("❌ Failed to log search execution:", error);
    // Don't throw - logging failure shouldn't break the search
  }
}

// Shared function for ranking and seeking clips
async function processClipsWithRankingAndSeeking(
  clipsForRanking: ClipForRanking[],
  query: string,
  topK: number,
): Promise<z.infer<typeof SearchSegment>[]> {
  if (clipsForRanking.length === 0) {
    console.log("⚠️ No clips prepared for ranking");
    return [];
  }

  // Step 1: Rank clips using the ranking prompt
  console.log("🏆 Step 1: Ranking clips using LLM...");
  const clipsForRankingPrompt = clipsForRanking.map((clip, index) => ({
    index,
    title: clip.videoTitle,
    videoId: clip.video_id,
    text: clip.transcriptText.slice(0, 300),
    startTime: clip.startTime,
    endTime: clip.endTime,
  }));

  const rankingPrompt = replacePromptPlaceholders(RANK_PROMPT, {
    query: query,
    clips: JSON.stringify(clipsForRankingPrompt, null, 2),
    clipCount: (clipsForRanking.length - 1).toString(),
    topK: topK.toString(),
  });

  console.log("🤖 Step 1: Calling GPT-4 for ranking...");
  const rankingCompletion = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    messages: [
      {
        role: "system",
        content: `You are an expert at understanding user intent and selecting the most relevant podcast segments. You must respond with a JSON object containing a 'rankings' array. Each ranking object must have 'clipIndex' (integer) and 'reasoning' (string) fields. CRITICAL: Ensure no overlapping clips from the same video are included in your rankings.`,
      },
      {
        role: "user",
        content: rankingPrompt,
      },
    ],
    temperature: 0.3,
    max_tokens: 500,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "ranking_response",
        strict: true,
        schema: {
          type: "object",
          properties: {
            rankings: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  clipIndex: {
                    type: "integer",
                    description:
                      "Index of the clip in the original list (0-based)",
                  },
                  reasoning: {
                    type: "string",
                    description:
                      "Brief explanation of why this clip is ranked at this position",
                  },
                },
                required: ["clipIndex", "reasoning"],
                additionalProperties: false,
              },
            },
          },
          required: ["rankings"],
          additionalProperties: false,
        },
      },
    },
  });

  const rankingResponse = rankingCompletion.choices[0]?.message?.content;
  if (!rankingResponse) {
    throw new Error("No response from ranking model");
  }

  // Save ranking prompt and response
  savePromptResponse(`ranking`, rankingPrompt, rankingResponse, {
    query: query,
    clipCount: clipsForRanking.length,
    topK: topK,
  });

  console.log("📊 Step 1: Ranking response:", rankingResponse);

  // Parse and validate ranking response
  let rankingResults: RankingOutput;
  try {
    const parsedResponse = JSON.parse(rankingResponse) as {
      rankings: RankingOutput;
    };
    // Handle both direct array and wrapped object formats
    const rankingsArray = parsedResponse.rankings || parsedResponse;
    rankingResults = RankingOutputSchema.parse(rankingsArray);
  } catch (error) {
    console.error("❌ Step 1: Failed to parse ranking response:", error);
    throw new Error("Invalid ranking response format");
  }

  console.log("📊 Step 1: Validated ranking results:", rankingResults);

  const topClips = rankingResults
    .slice(0, topK)
    .map((result) => clipsForRanking[result.clipIndex])
    .filter(
      (clip): clip is NonNullable<typeof clip> =>
        clip !== null && clip !== undefined,
    );

  console.log(
    `✅ Step 1: Selected ${topClips.length} top clips for precise seeking`,
  );

  // Step 2: Find precise starting points for each top clip
  console.log(
    `🎯 Step 2: Finding precise starting points for ${topClips.length} clips...`,
  );
  const finalSegments: z.infer<typeof SearchSegment>[] = [];

  // Process all clips in parallel
  const clipProcessingPromises = topClips.map(async (clip, i) => {
    if (!clip) return null;

    console.log(
      `🎯 Step 2.${i + 1}: Processing clip "${clip.videoTitle}" (${clip.timestampReadable})`,
    );

    try {
      // Extract word-level timestamps from transcript
      const transcriptSegments = Array.isArray(clip.transcriptData.segments)
        ? clip.transcriptData.segments
        : (JSON.parse(
            clip.transcriptData.segments as string,
          ) as TranscriptSegment[]);

      const words = extractWordTimestamps(
        transcriptSegments,
        clip.startTime,
        clip.endTime,
      );

      // Convert to structured word input format
      const wordsInput: SeekingWordInput = words.map((word, index) => ({
        w: word.word,
        i: index,
      }));

      console.log(
        `📝 Step 2.${i + 1}: Extracted ${words.length} words from transcript`,
      );

      // Use seeking prompt to find precise starting point
      const seekingPrompt = replacePromptPlaceholders(SEEK_PROMPT, {
        query: query,
        clipTitle: clip.videoTitle,
        timestamp: clip.timestampReadable,
        transcriptWords: JSON.stringify(wordsInput, null, 2),
        duration: clip.duration.toString(),
        topic: clip.transcriptText.slice(0, 100),
      });

      console.log(`🤖 Step 2.${i + 1}: Calling GPT-4 for precise seeking...`);
      const seekCompletion = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        messages: [
          {
            role: "system",
            content: `You are a precise content locator for podcast clips. Always respond with valid JSON.`,
          },
          {
            role: "user",
            content: seekingPrompt,
          },
        ],
        temperature: 0.3,
        max_tokens: 200,
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "seeking_response",
            strict: true,
            schema: {
              type: "object",
              properties: {
                start_index: {
                  type: "integer",
                  description:
                    "Index of the word where the answer begins (0-based)",
                },
                reasoning: {
                  type: "string",
                  description:
                    "Explanation of why this starting point was chosen",
                },
              },
              required: ["start_index", "reasoning"],
              additionalProperties: false,
            },
          },
        },
      });

      const seekResponse = seekCompletion.choices[0]?.message?.content;
      if (!seekResponse) {
        throw new Error("No response from seeking model");
      }

      // Save seeking prompt and response
      savePromptResponse(`seeking_${i + 1}`, seekingPrompt, seekResponse, {
        query: query,
        clipTitle: clip.videoTitle,
        clipId: clip.clip_id,
        startTime: clip.startTime,
        endTime: clip.endTime,
        wordCount: words.length,
      });

      console.log(`📊 Step 2.${i + 1}: Seeking response:`, seekResponse);

      // Parse and validate seeking response
      let seekResult: SeekingOutput;
      try {
        const parsedResponse = JSON.parse(seekResponse) as SeekingOutput;
        seekResult = SeekingOutputSchema.parse(parsedResponse);
      } catch (error) {
        console.error(
          `❌ Step 2.${i + 1}: Failed to parse seeking response:`,
          error,
        );
        throw new Error("Invalid seeking response format");
      }

      // Calculate precise start time from word index
      let preciseStartTime = clip.startTime;
      if (
        seekResult.start_index >= 0 &&
        seekResult.start_index < words.length &&
        words[seekResult.start_index]
      ) {
        preciseStartTime = words[seekResult.start_index]!.start;
      }

      console.log(
        `⏰ Step 2.${i + 1}: Original start: ${clip.startTime}s, Precise start: ${preciseStartTime}s (word index: ${seekResult.start_index})`,
      );

      // Extract updated transcript text from precise start time to end time
      const updatedTranscriptText = words
        .slice(seekResult.start_index)
        .map((word) => word.word)
        .join(" ");

      console.log(
        `📝 Step 2.${i + 1}: Updated transcript from ${preciseStartTime}s: "${updatedTranscriptText.slice(0, 100)}..."`,
      );

      // Create final segment
      const finalSegment: z.infer<typeof SearchSegment> = {
        id: clip.clip_id,
        score: clip.similarityScore,
        youtubeId: clip.youtubeId,
        videoTitle: clip.videoTitle,
        segmentId: clip.segmentId,
        startTime: preciseStartTime,
        endTime: clip.endTime,
        duration: clip.endTime - preciseStartTime,
        timestampReadable: `${Math.floor(preciseStartTime / 60)}:${Math.floor(
          preciseStartTime % 60,
        )
          .toString()
          .padStart(2, "0")} - ${Math.floor(clip.endTime / 60)}:${Math.floor(
          clip.endTime % 60,
        )
          .toString()
          .padStart(2, "0")}`,
        transcriptText: updatedTranscriptText,
        primaryEmotion: clip.primaryEmotion,
        emotionScore: clip.emotionScore,
      };

      console.log(
        `✅ Step 2.${i + 1}: Successfully processed clip with reasoning: ${seekResult.reasoning}`,
      );

      return finalSegment;
    } catch (error) {
      console.error(
        `❌ Step 2.${i + 1}: Error processing clip ${clip.clip_id}:`,
        error,
      );
      console.log(`🔄 Step 2.${i + 1}: Falling back to original segment data`);
      // Fallback: use original segment data
      return {
        id: clip.clip_id,
        score: clip.similarityScore,
        youtubeId: clip.youtubeId,
        videoTitle: clip.videoTitle,
        segmentId: clip.segmentId,
        startTime: clip.startTime,
        endTime: clip.endTime,
        duration: clip.duration,
        timestampReadable: clip.timestampReadable,
        transcriptText: clip.transcriptText,
        primaryEmotion: clip.primaryEmotion,
        emotionScore: clip.emotionScore,
      };
    }
  });

  // Wait for all clips to be processed in parallel
  const processedSegments = await Promise.all(clipProcessingPromises);

  // Filter out null results and add to final segments
  finalSegments.push(
    ...processedSegments.filter(
      (segment): segment is z.infer<typeof SearchSegment> => segment !== null,
    ),
  );

  console.log(`🎉 Search ranking and seeking completed successfully!`);
  console.log("📊 Final results:", {
    totalSegments: finalSegments.length,
    segments: finalSegments.map((s) => ({
      id: s.id,
      title: s.videoTitle,
      startTime: s.startTime,
      score: s.score,
    })),
  });

  return finalSegments;
}

export const searchRouter = createTRPCRouter({
  // New endpoint to get video details by YouTube ID
  getVideoByYoutubeId: publicProcedure
    .input(
      z.object({
        youtubeId: z.string().min(1),
      }),
    )
    .query(async ({ input, ctx }) => {
      try {
        console.log(
          `🔍 Getting video details for YouTube ID: ${input.youtubeId}`,
        );

        const videoDetails = await ctx.db
          .select({
            id: videos.id,
            youtubeId: videos.youtubeId,
            title: videos.title,
            description: videos.description,
            duration: videos.duration,
            publishedAt: videos.publishedAt,
            thumbnailUrl: videos.thumbnailUrl,
            url: videos.url,
          })
          .from(videos)
          .where(eq(videos.youtubeId, input.youtubeId))
          .limit(1);

        if (videoDetails.length === 0) {
          throw new Error(`Video with YouTube ID ${input.youtubeId} not found`);
        }

        const video = videoDetails[0]!;
        console.log(`✅ Found video: "${video.title}" (ID: ${video.id})`);

        return {
          id: video.id,
          youtubeId: video.youtubeId,
          title: video.title,
          description: video.description,
          duration: video.duration,
          publishedAt: video.publishedAt,
          thumbnailUrl: video.thumbnailUrl,
          url: video.url,
        };
      } catch (error) {
        console.error("💥 getVideoByYoutubeId failed with error:", error);
        throw new Error("Failed to get video details");
      }
    }),

  // Unified search endpoint that handles both global and video-specific searches
  search: publicProcedure
    .input(
      z.object({
        query: z.string().min(1).max(500),
        topK: z.number().default(5),
        videoId: z.number().int().positive().optional(),
      }),
    )
    .query(async ({ input, ctx }) => {
      const startTime = Date.now();
      let errorMessage: string | undefined;

      try {
        console.log(
          `🔍 Search started for query: "${input.query}"${input.videoId ? ` in video ID: ${input.videoId}` : ""}`,
        );

        // Step 1: Get embedding for the search query
        console.log("📝 Step 1: Generating embedding for query...");
        const embeddingResponse = await openai.embeddings.create({
          model: "text-embedding-3-large",
          input: input.query,
        });

        const queryEmbedding = embeddingResponse.data[0]?.embedding;
        if (!queryEmbedding) {
          throw new Error("Failed to generate embedding for query");
        }
        console.log("✅ Step 1: Embedding generated successfully");

        // Step 2: Search Pinecone for segments
        console.log("🔍 Step 2: Searching Pinecone for segments...");
        const index = pinecone.Index(PINECONE_INDEX_NAME);

        const searchQuery: PineconeSearchQuery = {
          vector: queryEmbedding,
          topK: input.videoId ? input.topK * 2 : 15, // Get more results for video-specific search
          includeMetadata: true,
        };

        // Add filter if searching within a specific video
        if (input.videoId) {
          searchQuery.filter = {
            video_id: { $eq: input.videoId },
          };
        }

        const searchResults = await index.query(searchQuery);

        if (!searchResults.matches || searchResults.matches.length === 0) {
          console.log("⚠️ Step 2: No matches found in Pinecone");
          return {
            segments: [],
            totalFound: 0,
            query: input.query,
          };
        }
        console.log(
          `✅ Step 2: Found ${searchResults.matches.length} matches in Pinecone`,
        );

        // Step 3: Extract video IDs and fetch transcripts
        console.log(
          "🎬 Step 3: Extracting video IDs and fetching transcripts...",
        );

        // Extract video IDs from search results
        const videoIds = new Set<number>();
        searchResults.matches.forEach((match) => {
          if (!match.metadata) return;
          const metadata = match.metadata as unknown as PineconeMetadata;
          videoIds.add(metadata.video_id);
        });

        // Fetch video details and transcripts
        const videoDetails = await ctx.db
          .select({
            youtubeId: videos.youtubeId,
            title: videos.title,
            id: videos.id,
            transcriptId: videos.transcriptId,
            transcriptSegments: transcripts.segments,
          })
          .from(videos)
          .leftJoin(transcripts, eq(videos.transcriptId, transcripts.id))
          .where(inArray(videos.id, Array.from(videoIds)));

        console.log(
          `✅ Step 3: Fetched ${videoDetails.length} video details with transcripts`,
        );

        const videoMap = new Map(videoDetails.map((v) => [v.id, v]));

        // Step 4: Prepare clips for ranking
        console.log("📋 Step 4: Preparing clips for ranking...");
        const clipsForRanking = searchResults.matches
          .map((match) => {
            if (!match.metadata) return null;

            const metadata = match.metadata as unknown as PineconeMetadata;
            const videoId = metadata.video_id;
            const video = videoMap.get(videoId);

            if (!video) return null;

            // Extract segment text using timestamp-based filtering
            const segmentText = getClipTranscriptByTimestamp(
              { segments: video.transcriptSegments },
              metadata.start_time,
              metadata.end_time,
            );

            return {
              clip_id: match.id ?? "",
              video_id: videoId,
              youtubeId: video.youtubeId,
              videoTitle: video.title,
              segmentId: metadata.segment_id,
              startTime: metadata.start_time,
              endTime: metadata.end_time,
              duration: metadata.duration,
              timestampReadable: metadata.timestamp_readable,
              transcriptText: segmentText,
              primaryEmotion: metadata.primary_emotion,
              emotionScore: metadata.primary_emotion_score,
              similarityScore: match.score ?? 0,
              transcriptData: {
                segments: video.transcriptSegments as TranscriptSegment[],
              },
            };
          })
          .filter((clip): clip is NonNullable<typeof clip> => clip !== null);

        console.log(
          `✅ Step 4: Prepared ${clipsForRanking.length} clips for ranking`,
        );

        // Step 5: Use shared function for ranking and seeking
        const finalSegments = await processClipsWithRankingAndSeeking(
          clipsForRanking,
          input.query,
          input.topK,
        );

        // Prepare response
        const response: SearchResponse = {
          segments: finalSegments,
          totalFound: finalSegments.length,
          query: input.query,
        };

        // Add videoInfo if we only have one video in the results
        if (videoDetails.length === 1) {
          const video = videoDetails[0]!;
          response.videoInfo = {
            id: video.id,
            youtubeId: video.youtubeId,
            title: video.title,
          };
        }

        const processingTimeMs = Date.now() - startTime;

        // Prepare metadata for logging
        const inputClipsMetadata = clipsForRanking.map((clip) => ({
          videoTitle: clip.videoTitle,
          startTime: clip.startTime,
          endTime: clip.endTime,
          transcriptText: clip.transcriptText.slice(0, 200), // Truncate for storage
          similarityScore: clip.similarityScore,
        }));

        const outputSegmentsMetadata = finalSegments.map((segment) => ({
          videoTitle: segment.videoTitle,
          startTime: segment.startTime,
          endTime: segment.endTime,
          transcriptText: segment.transcriptText.slice(0, 200), // Truncate for storage
          score: segment.score,
        }));

        // Log successful search execution
        await logSearchExecution(
          ctx,
          input.query,
          input.videoId,
          input.topK,
          clipsForRanking.length,
          finalSegments.length,
          inputClipsMetadata,
          outputSegmentsMetadata,
          processingTimeMs,
          "success",
        );

        return response;
      } catch (error) {
        const processingTimeMs = Date.now() - startTime;
        errorMessage = error instanceof Error ? error.message : "Unknown error";

        console.error("💥 Search failed with error:", error);

        // Log failed search execution
        await logSearchExecution(
          ctx,
          input.query,
          input.videoId,
          input.topK,
          0, // No input clips processed
          0, // No output segments
          [], // No input metadata
          [], // No output metadata
          processingTimeMs,
          "error",
          errorMessage,
        );

        throw new Error("Failed to perform search");
      }
    }),
});
