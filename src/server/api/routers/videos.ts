import { desc, eq } from 'drizzle-orm';
import { z } from 'zod';
import { createTRPCRouter, publicProcedure } from '~/server/api/trpc';
import { db } from '~/server/db';
import { videos } from '~/server/db/schema';

export const videosRouter = createTRPCRouter({
  // Get all videos with optional filtering
  getAll: publicProcedure
    .input(
      z.object({
        limit: z.number().min(1).max(1000).default(100),
        offset: z.number().min(0).default(0),
        status: z.string().optional(),
        playlistId: z.number().optional(),
      }),
    )
    .query(async ({ input }) => {
      let query = db
        .select({
          id: videos.id,
          youtubeId: videos.youtubeId,
          title: videos.title,
          description: videos.description,
          duration: videos.duration,
          publishedAt: videos.publishedAt,
          thumbnailUrl: videos.thumbnailUrl,
          url: videos.url,
          status: videos.status,
          createdAt: videos.createdAt,
          updatedAt: videos.updatedAt,
        })
        .from(videos);

      // Apply filters
      if (input.status) {
        query = query.where(eq(videos.status, input.status));
      }
      if (input.playlistId) {
        query = query.where(eq(videos.playlistId, input.playlistId));
      }

      const allVideos = await query
        .orderBy(desc(videos.createdAt))
        .limit(input.limit)
        .offset(input.offset);

      return allVideos;
    }),

  // Get a specific video by ID
  getById: publicProcedure
    .input(z.object({ id: z.number() }))
    .query(async ({ input }) => {
      const video = await db
        .select({
          id: videos.id,
          youtubeId: videos.youtubeId,
          title: videos.title,
          description: videos.description,
          duration: videos.duration,
          publishedAt: videos.publishedAt,
          thumbnailUrl: videos.thumbnailUrl,
          url: videos.url,
          status: videos.status,
          createdAt: videos.createdAt,
          updatedAt: videos.updatedAt,
        })
        .from(videos)
        .where(eq(videos.id, input.id))
        .limit(1);

      return video[0] || null;
    }),

  // Get videos with chapters for network visualization
  getWithChapters: publicProcedure
    .input(
      z.object({
        limit: z.number().min(1).max(100).default(50),
        offset: z.number().min(0).default(0),
      }),
    )
    .query(async ({ input }) => {
      const videosWithChapters = await db
        .select({
          id: videos.id,
          youtubeId: videos.youtubeId,
          title: videos.title,
          status: videos.status,
          createdAt: videos.createdAt,
        })
        .from(videos)
        .where(eq(videos.status, "finished")) // Only videos that have been processed
        .orderBy(desc(videos.createdAt))
        .limit(input.limit)
        .offset(input.offset);

      return videosWithChapters;
    }),

  // Get video statistics
  getStats: publicProcedure.query(async () => {
    const [totalVideos, finishedVideos, pendingVideos] = await Promise.all([
      db
        .select()
        .from(videos)
        .then((r) => r.length),
      db
        .select()
        .from(videos)
        .where(eq(videos.status, "finished"))
        .then((r) => r.length),
      db
        .select()
        .from(videos)
        .where(eq(videos.status, "pending"))
        .then((r) => r.length),
    ]);

    return {
      totalVideos,
      finishedVideos,
      pendingVideos,
    };
  }),
});
