import { eq, inArray, sql } from "drizzle-orm";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import OpenAI from "openai";
import { join } from "path";
import { z } from "zod";
import { env } from "~/env";
import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { transcripts, videos } from "~/server/db/schema";

import { Pinecone } from "@pinecone-database/pinecone";

// Initialize clients
const pinecone = new Pinecone({
  apiKey: env.PINECONE_API_KEY,
});

const openai = new OpenAI({
  apiKey: env.OPENAI_API_KEY,
});

const PINECONE_INDEX_NAME = "video-segments";

// Load prompts at build time
const RANK_PROMPT = readFileSync(
  join(process.cwd(), "src", "prompts", "rank_relevance.txt"),
  "utf-8",
);
const SEEK_PROMPT = readFileSync(
  join(process.cwd(), "src", "prompts", "seek_clip.txt"),
  "utf-8",
);

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
interface RankingClipInput {
  index: number;
  title: string;
  text: string;
  timestamp: string;
  emotion?: string;
  score: number;
}

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

// Types for search results (defined for potential future use)

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
  transcriptSegments: any[],
  startTime: number,
  endTime: number,
): Array<{ word: string; start: number; end: number }> {
  const words: Array<{ word: string; start: number; end: number }> = [];

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
  transcriptData: any,
  startTime: number,
  endTime: number,
): any[] {
  if (!transcriptData?.segments) return [];

  const segments = Array.isArray(transcriptData.segments)
    ? transcriptData.segments
    : JSON.parse(transcriptData.segments as string);

  // Find transcript segments that overlap with the given time range
  const overlappingSegments = segments.filter((seg: any) => {
    const segStart = seg.start || 0;
    const segEnd = seg.end || segStart + 1;

    // Check for overlap with some tolerance (±1 second)
    const tolerance = 1.0;
    return segStart - tolerance <= endTime && segEnd + tolerance >= startTime;
  });

  // Sort by start time
  overlappingSegments.sort((a: any, b: any) => (a.start || 0) - (b.start || 0));
  return overlappingSegments;
}

// Helper function to join transcript segments into text
function joinTranscriptSegments(segments: any[]): string {
  return segments
    .map((seg: any) => seg.text || "")
    .join(" ")
    .trim();
}

// Helper function to get clip transcript from video transcript using timestamp ranges (legacy function)
function getClipTranscriptByTimestamp(
  transcriptData: any,
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

// Helper function to replace placeholders in prompts
function replacePromptPlaceholders(
  prompt: string,
  replacements: Record<string, string>,
): string {
  let result = prompt;
  for (const [placeholder, value] of Object.entries(replacements)) {
    result = result.replace(
      new RegExp(`\\{\\{${placeholder}\\}\\}`, "g"),
      value,
    );
  }
  return result;
}

// Helper function to save prompts and responses to filesystem
function savePromptResponse(
  step: string,
  prompt: string,
  response: string,
  metadata?: Record<string, any>,
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

export const searchRouter = createTRPCRouter({
  // New smart search endpoint that implements two-stage retrieval
  smartSearch: publicProcedure
    .input(
      z.object({
        query: z.string().min(1).max(500),
        topK: z.number().default(5),
      }),
    )
    .query(async ({ input, ctx }) => {
      try {
        console.log("🔍 SmartSearch started for query:", input.query);

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
        console.log(
          "✅ Step 1: Embedding generated successfully (length:",
          queryEmbedding.length,
          ")",
        );

        // Step 2: Search Pinecone for top 15 segments
        console.log("🔍 Step 2: Searching Pinecone for top 15 segments...");
        const index = pinecone.Index(PINECONE_INDEX_NAME);
        const searchResults = await index.query({
          vector: queryEmbedding,
          topK: 15,
          includeMetadata: true,
        });

        if (!searchResults.matches || searchResults.matches.length === 0) {
          console.log("⚠️ Step 2: No matches found in Pinecone");
          return { segments: [], totalFound: 0 };
        }
        console.log(
          "✅ Step 2: Found",
          searchResults.matches.length,
          "matches in Pinecone",
        );
        console.log(searchResults.matches);

        // Step 3: Extract video IDs and timestamp ranges, then fetch transcripts for timestamp-based filtering
        console.log("🎬 Step 3: Extracting video IDs and timestamp ranges...");

        // Create a map of video_id -> timestamp ranges for efficient querying
        const videoTimestampMap = new Map<
          number,
          Array<{ startTime: number; endTime: number }>
        >();
        const videoIds = new Set<number>();

        searchResults.matches.forEach((match) => {
          if (!match.metadata) return;

          const metadata = match.metadata as unknown as PineconeMetadata;
          const videoId = metadata.video_id;
          const startTime = metadata.start_time;
          const endTime = metadata.end_time;

          videoIds.add(videoId);
          if (!videoTimestampMap.has(videoId)) {
            videoTimestampMap.set(videoId, []);
          }
          videoTimestampMap.get(videoId)!.push({ startTime, endTime });
        });

        console.log(
          "📊 Step 3: Found",
          videoIds.size,
          "unique video IDs with",
          Array.from(videoTimestampMap.values()).reduce(
            (sum, ranges) => sum + ranges.length,
            0,
          ),
          "total timestamp ranges needed",
        );

        // Pretty print the videoTimestampMap for easier debugging
        console.log(
          "📊 Step 3: Timestamp filtering map:\n" +
            Array.from(videoTimestampMap.entries())
              .map(
                ([videoId, timestampRanges]) =>
                  `  Video ID ${videoId}: ${timestampRanges
                    .map((range) => `${range.startTime}s-${range.endTime}s`)
                    .join(", ")}`,
              )
              .join("\n"),
        );

        console.log(
          "🔍 Step 3: Executing database query with full transcripts for timestamp filtering...",
        );
        console.log("📊 Step 3: Video IDs to fetch:", Array.from(videoIds));

        // Fetch full transcripts using Drizzle ORM
        const videoDetailsWithTranscripts = await ctx.db
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

        // Transform the results to match our expected format
        const videoDetails = videoDetailsWithTranscripts.map((row) => ({
          youtubeId: row.youtubeId,
          title: row.title,
          id: row.id,
          transcriptId: row.transcriptId,
          transcriptSegments: row.transcriptSegments,
        }));

        const videoMap = new Map(videoDetails.map((v) => [v.id, v]));
        console.log(
          "✅ Step 3: Fetched",
          videoDetails.length,
          "video details with transcripts",
        );

        // Type assertion for video details
        type VideoWithTranscript = {
          youtubeId: string;
          title: string;
          id: number;
          transcriptId: number | null;
          transcriptSegments: any;
        };

        // Step 5: Prepare clips for ranking (segments already filtered at database level)
        console.log(
          "📋 Step 5: Preparing clips for ranking (database-filtered segments)...",
        );
        const clipsForRanking = searchResults.matches
          .map((match) => {
            if (!match.metadata) return null;

            const metadata = match.metadata as unknown as PineconeMetadata;
            const videoId = metadata.video_id;
            const video = videoMap.get(videoId) as
              | VideoWithTranscript
              | undefined;

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
              transcriptData: { segments: video.transcriptSegments }, // Already filtered at DB level
            };
          })
          .filter((clip): clip is NonNullable<typeof clip> => clip !== null);

        console.log(
          "✅ Step 5: Prepared",
          clipsForRanking.length,
          "clips for ranking (database-filtered segments)",
        );

        // Log timestamp-based filtering benefits
        const totalTimestampRanges = Array.from(
          videoTimestampMap.values(),
        ).reduce((sum, ranges) => sum + ranges.length, 0);
        console.log(
          "📊 Timestamp-based Filtering: Processing",
          totalTimestampRanges,
          "timestamp ranges for transcript segment extraction",
        );

        if (clipsForRanking.length === 0) {
          console.log("⚠️ Step 5: No clips prepared for ranking");
          return { segments: [], totalFound: 0 };
        }

        // Step 6: Rank clips using the ranking prompt
        console.log("🏆 Step 6: Ranking clips using LLM...");
        const clipsForRankingPrompt: RankingClipInput[] = clipsForRanking.map(
          (clip, index) => ({
            index,
            title: clip.videoTitle,
            text: clip.transcriptText.slice(0, 300),
            timestamp: clip.timestampReadable,
            emotion: clip.primaryEmotion,
            score: clip.similarityScore,
          }),
        );

        const rankingPrompt = replacePromptPlaceholders(RANK_PROMPT, {
          query: input.query,
          clips: JSON.stringify(clipsForRankingPrompt, null, 2),
          clipCount: (clipsForRanking.length - 1).toString(),
          topK: input.topK.toString(),
        });

        console.log("🤖 Step 6: Calling GPT-4 for ranking...");
        const rankingCompletion = await openai.chat.completions.create({
          model: "gpt-4.1-mini",
          messages: [
            {
              role: "system",
              content:
                "You are an expert at understanding user intent and selecting the most relevant podcast segments. You must respond with a JSON object containing a 'rankings' array. Each ranking object must have 'clipIndex' (integer) and 'reasoning' (string) fields.",
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
        savePromptResponse("ranking", rankingPrompt, rankingResponse, {
          query: input.query,
          clipCount: clipsForRanking.length,
          topK: input.topK,
        });

        console.log("📊 Step 6: Ranking response:", rankingResponse);

        // Parse and validate ranking response
        let rankingResults: RankingOutput;
        try {
          const parsedResponse = JSON.parse(rankingResponse);
          // Handle both direct array and wrapped object formats
          const rankingsArray = parsedResponse.rankings || parsedResponse;
          rankingResults = RankingOutputSchema.parse(rankingsArray);
        } catch (error) {
          console.error("❌ Step 6: Failed to parse ranking response:", error);
          throw new Error("Invalid ranking response format");
        }

        console.log("📊 Step 6: Validated ranking results:", rankingResults);

        const topClips = rankingResults
          .slice(0, input.topK)
          .map((result) => clipsForRanking[result.clipIndex])
          .filter(
            (clip): clip is NonNullable<typeof clip> =>
              clip !== null && clip !== undefined,
          );

        console.log(
          "✅ Step 6: Selected",
          topClips.length,
          "top clips for precise seeking",
        );

        // Step 7: Find precise starting points for each top clip
        console.log(
          "🎯 Step 7: Finding precise starting points for",
          topClips.length,
          "clips...",
        );
        const finalSegments: z.infer<typeof SearchSegment>[] = [];

        // Process all clips in parallel
        const clipProcessingPromises = topClips.map(async (clip, i) => {
          if (!clip) return null;

          console.log(
            `🎯 Step 7.${i + 1}: Processing clip "${clip.videoTitle}" (${clip.timestampReadable})`,
          );

          try {
            // Extract word-level timestamps from transcript
            const transcriptSegments = Array.isArray(
              clip.transcriptData.segments,
            )
              ? clip.transcriptData.segments
              : JSON.parse(clip.transcriptData.segments as string);

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
              `📝 Step 7.${i + 1}: Extracted`,
              words.length,
              "words from transcript",
            );

            // Use seeking prompt to find precise starting point
            const seekingPrompt = replacePromptPlaceholders(SEEK_PROMPT, {
              query: input.query,
              clipTitle: clip.videoTitle,
              timestamp: clip.timestampReadable,
              transcriptWords: JSON.stringify(wordsInput, null, 2),
              duration: clip.duration.toString(),
              topic: clip.transcriptText.slice(0, 100),
            });

            console.log(
              `🤖 Step 7.${i + 1}: Calling GPT-4 for precise seeking...`,
            );
            const seekCompletion = await openai.chat.completions.create({
              model: "gpt-4.1-mini",
              messages: [
                {
                  role: "system",
                  content:
                    "You are a precise content locator for podcast clips. Always respond with valid JSON.",
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
            savePromptResponse(
              `seeking_${i + 1}`,
              seekingPrompt,
              seekResponse,
              {
                query: input.query,
                clipTitle: clip.videoTitle,
                clipId: clip.clip_id,
                startTime: clip.startTime,
                endTime: clip.endTime,
                wordCount: words.length,
              },
            );

            console.log(`📊 Step 7.${i + 1}: Seeking response:`, seekResponse);

            // Parse and validate seeking response
            let seekResult: SeekingOutput;
            try {
              const parsedResponse = JSON.parse(seekResponse);
              seekResult = SeekingOutputSchema.parse(parsedResponse);
            } catch (error) {
              console.error(
                `❌ Step 7.${i + 1}: Failed to parse seeking response:`,
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
              `⏰ Step 7.${i + 1}: Original start: ${clip.startTime}s, Precise start: ${preciseStartTime}s (word index: ${seekResult.start_index})`,
            );

            // Extract updated transcript text from precise start time to end time
            // const updatedTranscriptText = getClipTranscriptByTimestamp(
            //   { segments: transcriptSegments },
            //   preciseStartTime,
            //   clip.endTime,
            // );

            const updatedTranscriptText = words
              .slice(seekResult.start_index)
              .map((word) => word.word)
              .join(" ");

            console.log(
              `📝 Step 7.${i + 1}: Updated transcript from ${preciseStartTime}s: "${updatedTranscriptText.slice(0, 100)}..."`,
            );

            // Create final segment
            const finalSegment: z.infer<typeof SearchSegment> = {
              id: clip.clip_id,
              score: clip.similarityScore, // Use original similarity score since confidence was removed
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
                .padStart(
                  2,
                  "0",
                )} - ${Math.floor(clip.endTime / 60)}:${Math.floor(
                clip.endTime % 60,
              )
                .toString()
                .padStart(2, "0")}`,
              transcriptText: updatedTranscriptText,
              primaryEmotion: clip.primaryEmotion,
              emotionScore: clip.emotionScore,
            };

            console.log(
              `✅ Step 7.${i + 1}: Successfully processed clip with reasoning: ${seekResult.reasoning}`,
            );

            return finalSegment;
          } catch (error) {
            console.error(
              `❌ Step 7.${i + 1}: Error processing clip ${clip.clip_id}:`,
              error,
            );
            console.log(
              `🔄 Step 7.${i + 1}: Falling back to original segment data`,
            );
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
            (segment): segment is z.infer<typeof SearchSegment> =>
              segment !== null,
          ),
        );

        console.log("🎉 SmartSearch completed successfully!");
        console.log("📊 Final results:", {
          totalSegments: finalSegments.length,
          segments: finalSegments.map((s) => ({
            id: s.id,
            title: s.videoTitle,
            startTime: s.startTime,
            score: s.score,
          })),
        });

        return {
          segments: finalSegments,
          totalFound: finalSegments.length,
          query: input.query,
        };
      } catch (error) {
        console.error("💥 SmartSearch failed with error:", error);
        throw new Error("Failed to perform smart search");
      }
    }),
});
