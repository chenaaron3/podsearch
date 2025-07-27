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
        const clipsForRankingPrompt = clipsForRanking.map((clip, index) => ({
          index,
          title: clip.videoTitle,
          text: clip.transcriptText.slice(0, 300),
          timestamp: clip.timestampReadable,
          emotion: clip.primaryEmotion,
          score: clip.similarityScore,
        }));

        const clipsText = clipsForRankingPrompt
          .map(
            (clip, i) =>
              `${i}. "${clip.title}" (${clip.timestamp}, Score: ${clip.score.toFixed(3)}): ${clip.text}... ${clip.emotion ? `[${clip.emotion}]` : ""}`,
          )
          .join("\n\n");

        const rankingPrompt = replacePromptPlaceholders(RANK_PROMPT, {
          query: input.query,
          clips: clipsText,
          clipCount: (clipsForRanking.length - 1).toString(),
          topK: input.topK.toString(),
        });

        console.log("🤖 Step 6: Calling GPT-4 for ranking...");
        const rankingCompletion = await openai.chat.completions.create({
          model: "gpt-4",
          messages: [
            {
              role: "system",
              content:
                "You are an expert at understanding user intent and selecting the most relevant podcast segments. Always respond with a valid JSON array of numbers.",
            },
            {
              role: "user",
              content: rankingPrompt,
            },
          ],
          temperature: 0.3,
          max_tokens: 100,
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
        const selectedIndices = JSON.parse(rankingResponse) as number[];
        console.log("📊 Step 6: Selected indices:", selectedIndices);

        const topClips = selectedIndices
          .filter((index) => index >= 0 && index < clipsForRanking.length)
          .slice(0, input.topK)
          .map((index) => clipsForRanking[index])
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

        for (let i = 0; i < topClips.length; i++) {
          const clip = topClips[i];
          if (!clip) continue;

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
            const wordsText = words.map((w) => w.word).join(" ");
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
              transcriptWords: wordsText,
              duration: clip.duration.toString(),
              topic: clip.transcriptText.slice(0, 100),
            });

            console.log(
              `🤖 Step 7.${i + 1}: Calling GPT-4 for precise seeking...`,
            );
            const seekCompletion = await openai.chat.completions.create({
              model: "gpt-4",
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
            const seekResult = JSON.parse(seekResponse) as {
              start_index: number;
              confidence: number;
              reasoning: string;
              context_needed: boolean;
              key_phrase: string;
            };

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

            // Create final segment
            finalSegments.push({
              id: clip.clip_id,
              score: seekResult.confidence,
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
              transcriptText: clip.transcriptText,
              primaryEmotion: clip.primaryEmotion,
              emotionScore: clip.emotionScore,
            });

            console.log(
              `✅ Step 7.${i + 1}: Successfully processed clip with confidence ${seekResult.confidence}`,
            );
          } catch (error) {
            console.error(
              `❌ Step 7.${i + 1}: Error processing clip ${clip.clip_id}:`,
              error,
            );
            console.log(
              `🔄 Step 7.${i + 1}: Falling back to original segment data`,
            );
            // Fallback: use original segment data
            finalSegments.push({
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
            });
          }
        }

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

  // Step 1: Broad search - returns 15 segments from Pinecone
  broadSearch: publicProcedure
    .input(
      z.object({
        query: z.string().min(1).max(500),
        topK: z.number().default(15),
      }),
    )
    .query(async ({ input, ctx }) => {
      try {
        // Get embedding for the search query
        const embeddingResponse = await openai.embeddings.create({
          model: "text-embedding-3-large",
          input: input.query,
        });

        const queryEmbedding = embeddingResponse.data[0]?.embedding;
        if (!queryEmbedding) {
          throw new Error("Failed to generate embedding for query");
        }

        // Search Pinecone
        const index = pinecone.Index(PINECONE_INDEX_NAME);
        const searchResults = await index.query({
          vector: queryEmbedding,
          topK: input.topK,
          includeMetadata: true,
        });

        if (!searchResults.matches) {
          return { segments: [], totalFound: 0 };
        }

        // Extract video IDs to fetch video details
        const videoIds = Array.from(
          new Set(
            searchResults.matches
              .map((match) => {
                const metadata = match.metadata as unknown as PineconeMetadata;
                const videoId = metadata?.video_id;
                return videoId || null;
              })
              .filter(Boolean),
          ),
        ) as number[];

        // Fetch video details from database using video IDs directly
        const videoDetails = await ctx.db
          .select({
            youtubeId: videos.youtubeId,
            title: videos.title,
            id: videos.id,
            transcriptId: videos.transcriptId,
          })
          .from(videos)
          .where(inArray(videos.id, videoIds));

        const videoMap = new Map(videoDetails.map((v) => [v.id, v]));

        // Get transcript IDs from videos that have transcripts
        const transcriptIds = videoDetails
          .map((v) => v.transcriptId)
          .filter(Boolean) as number[];

        // Get transcript segments for the matching videos
        const transcriptDetails = await ctx.db
          .select({
            id: transcripts.id,
            segments: transcripts.segments,
          })
          .from(transcripts)
          .where(inArray(transcripts.id, transcriptIds));

        const transcriptMap = new Map(transcriptDetails.map((t) => [t.id, t]));

        // Process and enrich search results
        const enrichedSegments: z.infer<typeof SearchSegment>[] =
          searchResults.matches
            .map((match) => {
              if (!match.metadata) return null;

              const metadata = match.metadata as unknown as PineconeMetadata;
              const videoId = metadata.video_id;

              const video = videoMap.get(videoId) as
                | {
                    youtubeId: string;
                    title: string;
                    id: number;
                    transcriptId: number | null;
                    transcriptSegments: any;
                  }
                | undefined;
              if (!video) return null;

              // Extract the specific segment text using timestamp-based filtering
              const segmentText = getClipTranscriptByTimestamp(
                { segments: video.transcriptSegments },
                metadata.start_time,
                metadata.end_time,
              );

              return {
                id: match.id ?? "",
                score: match.score ?? 0,
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
              };
            })
            .filter(Boolean) as z.infer<typeof SearchSegment>[];

        return {
          segments: enrichedSegments,
          totalFound: enrichedSegments.length,
          query: input.query,
        };
      } catch (error) {
        console.error("Error in broad search:", error);
        throw new Error("Failed to perform search");
      }
    }),

  // Step 2: Generate clarifying questions based on search results
  generateClarifyingQuestions: publicProcedure
    .input(
      z.object({
        originalQuery: z.string(),
        searchResults: z.array(SearchSegment),
      }),
    )
    .mutation(async ({ input }) => {
      try {
        // Analyze the search results to understand the diversity of topics
        const topics = input.searchResults.map((segment) => ({
          title: segment.videoTitle,
          text: segment.transcriptText.slice(0, 200), // First 200 chars
          emotion: segment.primaryEmotion,
          timestamp: segment.timestampReadable,
        }));

        const prompt = `Based on a user's search query "${input.originalQuery}" and these ${input.searchResults.length} podcast segments from "Diary of a CEO", generate 3-4 clarifying questions to help narrow down what the user is most interested in.

Search Results Summary:
${topics
  .map(
    (topic, i) =>
      `${i + 1}. "${topic.title}" (${topic.timestamp}): ${topic.text}... ${topic.emotion ? `[Emotion: ${topic.emotion}]` : ""}`,
  )
  .join("\n")}

Generate clarifying questions that help the user choose between different:
- Specific guests/episodes
- Topics or themes 
- Emotional tones or contexts
- Time periods or specific discussions

Format as a JSON array of strings. Questions should be conversational and help narrow down to the most relevant 5 segments.

Example format: ["Are you more interested in X or Y?", "Would you prefer segments about...", etc.]`;

        const completion = await openai.chat.completions.create({
          model: "gpt-4",
          messages: [
            {
              role: "system",
              content:
                "You are a helpful assistant that generates clarifying questions to help users find the most relevant podcast segments. Always respond with valid JSON.",
            },
            {
              role: "user",
              content: prompt,
            },
          ],
          temperature: 0.7,
          max_tokens: 500,
        });

        const responseContent = completion.choices[0]?.message?.content;
        if (!responseContent) {
          throw new Error("No response from OpenAI");
        }

        // Parse the JSON response
        const questions = JSON.parse(responseContent) as string[];

        return {
          questions,
          originalQuery: input.originalQuery,
          totalSegments: input.searchResults.length,
        };
      } catch (error) {
        console.error("Error generating clarifying questions:", error);
        // Fallback questions if AI fails
        return {
          questions: [
            "Are you looking for practical advice or personal stories?",
            "Would you prefer recent episodes or older content?",
            "Are you interested in specific industries or general business advice?",
          ],
          originalQuery: input.originalQuery,
          totalSegments: input.searchResults.length,
        };
      }
    }),

  // Step 3: Refined search based on clarifying answers
  refinedSearch: publicProcedure
    .input(
      z.object({
        originalQuery: z.string(),
        clarifyingAnswers: z.array(z.string()),
        originalResults: z.array(SearchSegment),
        targetCount: z.number().default(5),
      }),
    )
    .mutation(async ({ input }) => {
      try {
        // Use LLM to select and rank the best segments based on clarifying answers
        const resultsContext = input.originalResults.map((segment, i) => ({
          index: i,
          title: segment.videoTitle,
          text: segment.transcriptText.slice(0, 300),
          timestamp: segment.timestampReadable,
          emotion: segment.primaryEmotion,
          score: segment.score,
        }));

        const prompt = `Original search: "${input.originalQuery}"

User clarifications: ${input.clarifyingAnswers.join(". ")}

From these ${input.originalResults.length} podcast segments, select the top ${input.targetCount} most relevant ones based on the user's clarifications:

${resultsContext
  .map(
    (segment, i) =>
      `${i}. "${segment.title}" (${segment.timestamp}, Score: ${segment.score.toFixed(3)}): ${segment.text}... ${segment.emotion ? `[${segment.emotion}]` : ""}`,
  )
  .join("\n\n")}

Return only the indices (0-${input.originalResults.length - 1}) of the top ${input.targetCount} segments as a JSON array of numbers, ordered by relevance to the user's clarified intent.

Example: [2, 7, 1, 12, 4]`;

        const completion = await openai.chat.completions.create({
          model: "gpt-4",
          messages: [
            {
              role: "system",
              content:
                "You are an expert at understanding user intent and selecting the most relevant podcast segments. Always respond with a valid JSON array of numbers.",
            },
            {
              role: "user",
              content: prompt,
            },
          ],
          temperature: 0.3,
          max_tokens: 100,
        });

        const responseContent = completion.choices[0]?.message?.content;
        if (!responseContent) {
          throw new Error("No response from OpenAI");
        }

        const selectedIndices = JSON.parse(responseContent) as number[];

        // Validate indices and select segments
        const refinedSegments = selectedIndices
          .filter((index) => index >= 0 && index < input.originalResults.length)
          .slice(0, input.targetCount)
          .map((index) => input.originalResults[index])
          .filter(Boolean) as z.infer<typeof SearchSegment>[];

        return {
          segments: refinedSegments,
          originalQuery: input.originalQuery,
          clarifyingAnswers: input.clarifyingAnswers,
          selectionReasoning:
            "Segments selected based on user clarifications and relevance ranking",
        };
      } catch (error) {
        console.error("Error in refined search:", error);
        // Fallback: return top scoring segments
        const fallbackSegments = input.originalResults
          .sort((a, b) => b.score - a.score)
          .slice(0, input.targetCount);

        return {
          segments: fallbackSegments,
          originalQuery: input.originalQuery,
          clarifyingAnswers: input.clarifyingAnswers,
          selectionReasoning: "Fallback selection based on similarity scores",
        };
      }
    }),

  // Helper endpoint to get full transcript for a specific segment
  getSegmentTranscript: publicProcedure
    .input(
      z.object({
        youtubeId: z.string(),
        segmentId: z.number(),
      }),
    )
    .query(async ({ input, ctx }) => {
      try {
        // First find the video by youtubeId, then get its transcript
        const video = await ctx.db
          .select({
            transcriptId: videos.transcriptId,
          })
          .from(videos)
          .where(eq(videos.youtubeId, input.youtubeId))
          .limit(1);

        if (!video[0]?.transcriptId) {
          throw new Error("Video or transcript not found");
        }

        const transcript = await ctx.db
          .select({
            segments: transcripts.segments,
          })
          .from(transcripts)
          .where(eq(transcripts.id, video[0].transcriptId))
          .limit(1);

        if (!transcript[0]) {
          throw new Error("Transcript not found");
        }

        type SegmentType = { id?: number; segment_id?: number; text?: string };
        const segments = (
          Array.isArray(transcript[0].segments)
            ? transcript[0].segments
            : JSON.parse(transcript[0].segments as string)
        ) as SegmentType[];

        const segment = segments.find(
          (seg) =>
            seg.id === input.segmentId || seg.segment_id === input.segmentId,
        );

        return {
          segmentText: segment?.text ?? "",
          allSegments: segments,
        };
      } catch (error) {
        console.error("Error fetching segment transcript:", error);
        throw new Error("Failed to fetch transcript");
      }
    }),
});
