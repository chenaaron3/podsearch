import { eq, inArray } from "drizzle-orm";
import OpenAI from "openai";
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

export const searchRouter = createTRPCRouter({
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

              const video = videoMap.get(videoId);
              const youtubeId = video?.youtubeId || "";
              const transcript = (video as any)?.transcriptId
                ? transcriptMap.get((video as any).transcriptId)
                : null;

              if (!video || !transcript) return null;

              // Extract the specific segment text from transcript segments
              let segmentText = "";
              if ((transcript as any).segments) {
                type SegmentType = {
                  id?: number;
                  segment_id?: number;
                  text?: string;
                };
                const segments = (
                  Array.isArray((transcript as any).segments)
                    ? (transcript as any).segments
                    : JSON.parse((transcript as any).segments as string)
                ) as SegmentType[];

                const segmentId = metadata.segment_id;
                const matchingSegment = segments.find(
                  (seg) => seg.id === segmentId || seg.segment_id === segmentId,
                );
                segmentText = matchingSegment?.text ?? "";
              }

              return {
                id: match.id ?? "",
                score: match.score ?? 0,
                youtubeId,
                videoTitle: (video as any).title,
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
