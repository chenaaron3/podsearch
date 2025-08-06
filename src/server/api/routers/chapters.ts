import { and, desc, eq, gte, inArray, ne } from 'drizzle-orm';
import { z } from 'zod';
import { createTRPCRouter, publicProcedure } from '~/server/api/trpc';
import { db } from '~/server/db';
import { chapters, chapterSimilarities, videos } from '~/server/db/schema';

export const chaptersRouter = createTRPCRouter({
  // Get chapters for a specific video
  getChaptersByVideo: publicProcedure
    .input(z.object({ videoId: z.number() }))
    .query(async ({ input }) => {
      const videoChapters = await db
        .select()
        .from(chapters)
        .where(eq(chapters.videoId, input.videoId))
        .orderBy(chapters.chapterIdx);

      return videoChapters;
    }),

  // Get all chapters with video information
  getAllChapters: publicProcedure
    .input(
      z.object({
        limit: z.number().min(1).max(100).default(50),
        offset: z.number().min(0).default(0),
      }),
    )
    .query(async ({ input }) => {
      const allChapters = await db
        .select({
          id: chapters.id,
          chapterVideoId: chapters.videoId,
          chapterIdx: chapters.chapterIdx,
          chapterName: chapters.chapterName,
          chapterSummary: chapters.chapterSummary,
          startTime: chapters.startTime,
          endTime: chapters.endTime,
          createdAt: chapters.createdAt,
          updatedAt: chapters.updatedAt,
          videoId: videos.id,
          videoYoutubeId: videos.youtubeId,
          videoTitle: videos.title,
        })
        .from(chapters)
        .innerJoin(videos, eq(chapters.videoId, videos.id))
        .orderBy(desc(chapters.createdAt))
        .limit(input.limit)
        .offset(input.offset);

      return allChapters;
    }),

  // Get similar chapters for a specific chapter
  getSimilarChapters: publicProcedure
    .input(
      z.object({
        chapterId: z.number(),
        limit: z.number().min(1).max(20).default(5),
      }),
    )
    .query(async ({ input }) => {
      const similarities = await db
        .select({
          id: chapterSimilarities.id,
          sourceChapterId: chapterSimilarities.sourceChapterId,
          destChapterId: chapterSimilarities.destChapterId,
          similarityScore: chapterSimilarities.similarityScore,
          createdAt: chapterSimilarities.createdAt,
          destChapterDbId: chapters.id,
          destChapterVideoId: chapters.videoId,
          destChapterIdx: chapters.chapterIdx,
          destChapterName: chapters.chapterName,
          destChapterSummary: chapters.chapterSummary,
          destChapterStartTime: chapters.startTime,
          destChapterEndTime: chapters.endTime,
          destVideoId: videos.id,
          destVideoYoutubeId: videos.youtubeId,
          destVideoTitle: videos.title,
        })
        .from(chapterSimilarities)
        .innerJoin(chapters, eq(chapterSimilarities.destChapterId, chapters.id))
        .innerJoin(videos, eq(chapters.videoId, videos.id))
        .where(eq(chapterSimilarities.sourceChapterId, input.chapterId))
        .orderBy(desc(chapterSimilarities.similarityScore))
        .limit(input.limit);

      return similarities;
    }),

  // Get graph data for visualization
  getGraphData: publicProcedure
    .input(
      z.object({
        videoId: z.number().optional(),
        limit: z.number().min(1).max(100).default(50),
      }),
    )
    .query(async ({ input }) => {
      // Get chapters (filtered by video if specified)
      const baseQuery = db
        .select({
          id: chapters.id,
          chapterVideoId: chapters.videoId,
          chapterIdx: chapters.chapterIdx,
          chapterName: chapters.chapterName,
          chapterSummary: chapters.chapterSummary,
          startTime: chapters.startTime,
          endTime: chapters.endTime,
          videoId: videos.id,
          videoYoutubeId: videos.youtubeId,
          videoTitle: videos.title,
        })
        .from(chapters)
        .innerJoin(videos, eq(chapters.videoId, videos.id));

      const allChapters = input.videoId
        ? await baseQuery
            .where(eq(chapters.videoId, input.videoId))
            .limit(input.limit)
        : await baseQuery.limit(input.limit);

      // Get similarities for these chapters
      const chapterIds = allChapters.map((c) => c.id);
      const similarities = await db
        .select({
          sourceChapterId: chapterSimilarities.sourceChapterId,
          destChapterId: chapterSimilarities.destChapterId,
          similarityScore: chapterSimilarities.similarityScore,
        })
        .from(chapterSimilarities)
        .where(
          and(
            inArray(chapterSimilarities.sourceChapterId, chapterIds),
            inArray(chapterSimilarities.destChapterId, chapterIds),
          ),
        );

      // Convert to graph format
      const nodes = allChapters.map((chapter) => ({
        id: chapter.id.toString(),
        label: chapter.chapterName,
        videoId: chapter.videoId,
        videoTitle: chapter.videoTitle,
        chapterName: chapter.chapterName,
        chapterSummary: chapter.chapterSummary,
        startTime: chapter.startTime,
        endTime: chapter.endTime,
      }));

      const edges = similarities.map((similarity) => ({
        source: similarity.sourceChapterId.toString(),
        target: similarity.destChapterId.toString(),
        similarityScore: similarity.similarityScore,
      }));

      return {
        nodes,
        edges,
      };
    }),

  // Get chapter statistics
  getStats: publicProcedure.query(async () => {
    const [totalChapters, totalSimilarities, totalVideos] = await Promise.all([
      db
        .select()
        .from(chapters)
        .then((r) => r.length),
      db
        .select()
        .from(chapterSimilarities)
        .then((r) => r.length),
      db
        .select()
        .from(videos)
        .then((r) => r.length),
    ]);

    return {
      totalChapters,
      totalSimilarities,
      totalVideos,
    };
  }),

  // Get network visualization data with similarity threshold
  getNetworkData: publicProcedure
    .input(
      z.object({
        similarityThreshold: z.number().min(0).max(1).default(0.5),
        limit: z.number().min(1).max(1000).default(500),
        videoId: z.number().optional(),
      }),
    )
    .query(async ({ input }) => {
      // Get similarities above threshold
      const similarities = await db
        .select({
          sourceChapterId: chapterSimilarities.sourceChapterId,
          destChapterId: chapterSimilarities.destChapterId,
          similarityScore: chapterSimilarities.similarityScore,
        })
        .from(chapterSimilarities)
        .where(
          and(
            gte(chapterSimilarities.similarityScore, input.similarityThreshold),
            ne(
              chapterSimilarities.sourceChapterId,
              chapterSimilarities.destChapterId,
            ), // Exclude self-loops
          ),
        )
        .orderBy(desc(chapterSimilarities.similarityScore))
        .limit(input.limit);

      // Get unique chapter IDs from similarities
      const chapterIds = new Set<number>();
      similarities.forEach((s) => {
        chapterIds.add(s.sourceChapterId);
        chapterIds.add(s.destChapterId);
      });

      // Get chapter details
      const chaptersData = await db
        .select({
          id: chapters.id,
          chapterVideoId: chapters.videoId,
          chapterIdx: chapters.chapterIdx,
          chapterName: chapters.chapterName,
          chapterSummary: chapters.chapterSummary,
          startTime: chapters.startTime,
          endTime: chapters.endTime,
          videoId: videos.id,
          videoYoutubeId: videos.youtubeId,
          videoTitle: videos.title,
        })
        .from(chapters)
        .innerJoin(videos, eq(chapters.videoId, videos.id))
        .where(inArray(chapters.id, Array.from(chapterIds)));

      // Filter by video if specified
      const filteredChapters = input.videoId
        ? chaptersData.filter((c) => c.videoId === input.videoId)
        : chaptersData;

      // Filter similarities to only include chapters we have
      const filteredChapterIds = new Set(filteredChapters.map((c) => c.id));
      const filteredSimilarities = similarities.filter(
        (s) =>
          filteredChapterIds.has(s.sourceChapterId) &&
          filteredChapterIds.has(s.destChapterId),
      );

      // Convert to Cytoscape format
      const nodes = filteredChapters.map((chapter) => ({
        data: {
          id: chapter.id.toString(),
          label: chapter.chapterName,
          videoId: chapter.videoId,
          videoTitle: chapter.videoTitle,
          chapterName: chapter.chapterName,
          chapterSummary: chapter.chapterSummary,
          startTime: chapter.startTime,
          endTime: chapter.endTime,
          duration: chapter.endTime - chapter.startTime,
        },
      }));

      const edges = filteredSimilarities.map((similarity) => ({
        data: {
          id: `${similarity.sourceChapterId}-${similarity.destChapterId}`,
          source: similarity.sourceChapterId.toString(),
          target: similarity.destChapterId.toString(),
          similarityScore: similarity.similarityScore,
          weight: similarity.similarityScore,
        },
      }));

      return {
        nodes,
        edges,
        stats: {
          totalNodes: nodes.length,
          totalEdges: edges.length,
          averageSimilarity:
            edges.length > 0
              ? edges.reduce(
                  (sum, edge) => sum + edge.data.similarityScore,
                  0,
                ) / edges.length
              : 0,
        },
      };
    }),
});
