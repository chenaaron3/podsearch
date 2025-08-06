import { chaptersRouter } from '~/server/api/routers/chapters';
import { postRouter } from '~/server/api/routers/post';
import { searchRouter } from '~/server/api/routers/search';
import { videosRouter } from '~/server/api/routers/videos';
import { createCallerFactory, createTRPCRouter } from '~/server/api/trpc';

/**
 * This is the primary router for your server.
 *
 * All routers added in /api/routers should be manually added here.
 */
export const appRouter = createTRPCRouter({
  post: postRouter,
  search: searchRouter,
  chapters: chaptersRouter,
  videos: videosRouter,
});

// export type definition of API
export type AppRouter = typeof appRouter;

/**
 * Create a server-side caller for the tRPC API.
 * @example
 * const trpc = createCaller(createContext);
 * const res = await trpc.post.all();
 *       ^? Post[]
 */
export const createCaller = createCallerFactory(appRouter);
