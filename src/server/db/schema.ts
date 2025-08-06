import { relations, sql } from 'drizzle-orm';
import { index, pgEnum, pgTableCreator, primaryKey, unique } from 'drizzle-orm/pg-core';

import type { AdapterAccount } from "next-auth/adapters";
/**
 * This is an example of how to use the multi-project schema feature of Drizzle ORM. Use the same
 * database instance for multiple projects.
 *
 * @see https://orm.drizzle.team/docs/goodies#multi-project-schema
 */
export const createTable = pgTableCreator((name) => `podsearch_${name}`);

// Video processing statuses: pending, downloaded, processed, embedded, finished, failed

export const posts = createTable(
  "post",
  (d) => ({
    id: d.integer().primaryKey().generatedByDefaultAsIdentity(),
    name: d.varchar({ length: 256 }),
    createdById: d
      .varchar({ length: 255 })
      .notNull()
      .references(() => users.id),
    createdAt: d
      .timestamp({ withTimezone: true })
      .default(sql`CURRENT_TIMESTAMP`)
      .notNull(),
    updatedAt: d.timestamp({ withTimezone: true }).$onUpdate(() => new Date()),
  }),
  (t) => [
    index("created_by_idx").on(t.createdById),
    index("name_idx").on(t.name),
  ],
);

export const users = createTable("user", (d) => ({
  id: d
    .varchar({ length: 255 })
    .notNull()
    .primaryKey()
    .$defaultFn(() => crypto.randomUUID()),
  name: d.varchar({ length: 255 }),
  email: d.varchar({ length: 255 }).notNull(),
  emailVerified: d
    .timestamp({
      mode: "date",
      withTimezone: true,
    })
    .default(sql`CURRENT_TIMESTAMP`),
  image: d.varchar({ length: 255 }),
}));

export const usersRelations = relations(users, ({ many }) => ({
  accounts: many(accounts),
}));

export const accounts = createTable(
  "account",
  (d) => ({
    userId: d
      .varchar({ length: 255 })
      .notNull()
      .references(() => users.id),
    type: d.varchar({ length: 255 }).$type<AdapterAccount["type"]>().notNull(),
    provider: d.varchar({ length: 255 }).notNull(),
    providerAccountId: d.varchar({ length: 255 }).notNull(),
    refresh_token: d.text(),
    access_token: d.text(),
    expires_at: d.integer(),
    token_type: d.varchar({ length: 255 }),
    scope: d.varchar({ length: 255 }),
    id_token: d.text(),
    session_state: d.varchar({ length: 255 }),
  }),
  (t) => [
    primaryKey({ columns: [t.provider, t.providerAccountId] }),
    index("account_user_id_idx").on(t.userId),
  ],
);

export const accountsRelations = relations(accounts, ({ one }) => ({
  user: one(users, { fields: [accounts.userId], references: [users.id] }),
}));

export const sessions = createTable(
  "session",
  (d) => ({
    sessionToken: d.varchar({ length: 255 }).notNull().primaryKey(),
    userId: d
      .varchar({ length: 255 })
      .notNull()
      .references(() => users.id),
    expires: d.timestamp({ mode: "date", withTimezone: true }).notNull(),
  }),
  (t) => [index("t_user_id_idx").on(t.userId)],
);

export const sessionsRelations = relations(sessions, ({ one }) => ({
  user: one(users, { fields: [sessions.userId], references: [users.id] }),
}));

export const verificationTokens = createTable(
  "verification_token",
  (d) => ({
    identifier: d.varchar({ length: 255 }).notNull(),
    token: d.varchar({ length: 255 }).notNull(),
    expires: d.timestamp({ mode: "date", withTimezone: true }).notNull(),
  }),
  (t) => [primaryKey({ columns: [t.identifier, t.token] })],
);

// Video processing pipeline tables
export const playlists = createTable(
  "playlist",
  (d) => ({
    id: d.integer().primaryKey().generatedByDefaultAsIdentity(),
    youtubeId: d.varchar({ length: 255 }).notNull().unique(),
    title: d.varchar({ length: 500 }).notNull(),
    description: d.text(),
    channelName: d.varchar({ length: 255 }),
    channelId: d.varchar({ length: 255 }),
    url: d.varchar({ length: 500 }).notNull(),
    totalVideos: d.integer().default(0),
    createdAt: d
      .timestamp({ withTimezone: true })
      .default(sql`CURRENT_TIMESTAMP`)
      .notNull(),
    updatedAt: d.timestamp({ withTimezone: true }).$onUpdate(() => new Date()),
    lastSyncAt: d.timestamp({ withTimezone: true }),
  }),
  (t) => [
    index("playlist_youtube_id_idx").on(t.youtubeId),
    index("playlist_channel_id_idx").on(t.channelId),
  ],
);

export const videos = createTable(
  "video",
  (d) => ({
    id: d.integer().primaryKey().generatedByDefaultAsIdentity(),
    youtubeId: d.varchar({ length: 255 }).notNull().unique(),
    playlistId: d
      .integer()
      .references(() => playlists.id, { onDelete: "cascade" }),
    transcriptId: d
      .integer()
      .references(() => transcripts.id, { onDelete: "set null" }),
    title: d.varchar({ length: 500 }).notNull(),
    description: d.text(),
    duration: d.integer(), // duration in seconds
    publishedAt: d.timestamp({ withTimezone: true }),
    thumbnailUrl: d.varchar({ length: 500 }),
    url: d.varchar({ length: 500 }).notNull(),
    localFilePath: d.varchar({ length: 1000 }),
    status: d.varchar({ length: 20 }).default("pending").notNull(),
    processingStartedAt: d.timestamp({ withTimezone: true }),
    processingCompletedAt: d.timestamp({ withTimezone: true }),
    errorMessage: d.text(),
    retryCount: d.integer().default(0),
    createdAt: d
      .timestamp({ withTimezone: true })
      .default(sql`CURRENT_TIMESTAMP`)
      .notNull(),
    updatedAt: d.timestamp({ withTimezone: true }).$onUpdate(() => new Date()),
  }),
  (t) => [
    index("video_youtube_id_idx").on(t.youtubeId),
    index("video_playlist_id_idx").on(t.playlistId),
    index("video_transcript_id_idx").on(t.transcriptId),
    index("video_status_idx").on(t.status),
    index("video_published_at_idx").on(t.publishedAt),
  ],
);

// Transcripts table for storing transcript data
export const transcripts = createTable(
  "transcript",
  (d) => ({
    id: d.integer().primaryKey().generatedByDefaultAsIdentity(),
    language: d.varchar({ length: 10 }).default("en"),
    // Store the complete transcript data as JSON
    segments: d.jsonb(), // Array of transcript segments with timestamps
    processingMetadata: d.jsonb(), // Metadata about processing (model, version, etc.)
    createdAt: d
      .timestamp({ withTimezone: true })
      .default(sql`CURRENT_TIMESTAMP`)
      .notNull(),
    updatedAt: d.timestamp({ withTimezone: true }).$onUpdate(() => new Date()),
  }),
  (t) => [index("transcript_language_idx").on(t.language)],
);

export const playlistsRelations = relations(playlists, ({ many }) => ({
  videos: many(videos),
}));

export const videosRelations = relations(videos, ({ one, many }) => ({
  playlist: one(playlists, {
    fields: [videos.playlistId],
    references: [playlists.id],
  }),
  transcript: one(transcripts, {
    fields: [videos.transcriptId],
    references: [transcripts.id],
  }),
  chapters: many(chapters),
}));

export const transcriptsRelations = relations(transcripts, ({ one }) => ({
  video: one(videos, {
    fields: [transcripts.id],
    references: [videos.transcriptId],
  }),
}));

// Search execution logging table with 30-day TTL
export const searchExecutions = createTable(
  "search_execution",
  (d) => ({
    id: d.integer().primaryKey().generatedByDefaultAsIdentity(),
    userId: d
      .varchar({ length: 255 })
      .references(() => users.id, { onDelete: "set null" }),
    query: d.varchar({ length: 500 }).notNull(),
    videoId: d.integer().references(() => videos.id, { onDelete: "set null" }),
    topK: d.integer().default(5),
    inputClipsCount: d.integer().default(0),
    outputSegmentsCount: d.integer().default(0),
    inputClipsMetadata: d.jsonb(), // Array of clip metadata objects
    outputSegmentsMetadata: d.jsonb(), // Array of segment metadata objects
    processingTimeMs: d.integer(),
    status: d.varchar({ length: 20 }).default("success"), // 'success' | 'error'
    errorMessage: d.text(),
    createdAt: d
      .timestamp({ withTimezone: true })
      .default(sql`CURRENT_TIMESTAMP`)
      .notNull(),
  }),
  (t) => [
    index("search_execution_video_id_idx").on(t.videoId),
    index("search_execution_created_at_idx").on(t.createdAt),
    index("search_execution_status_idx").on(t.status),
  ],
);

export const searchExecutionsRelations = relations(
  searchExecutions,
  ({ one }) => ({
    user: one(users, {
      fields: [searchExecutions.userId],
      references: [users.id],
    }),
    video: one(videos, {
      fields: [searchExecutions.videoId],
      references: [videos.id],
    }),
  }),
);

// API usage tracking table for transcript requests
export const transcriptRequests = createTable(
  "transcript_request",
  (d) => ({
    id: d.integer().primaryKey().generatedByDefaultAsIdentity(),
    youtubeId: d.varchar({ length: 255 }).notNull(),
    timestamp: d.integer().notNull(), // timestamp in seconds
    duration: d.integer().notNull(), // duration in seconds
    transcriptText: d.text(), // the returned transcript text
    success: d.boolean().default(true), // whether the request was successful
    errorMessage: d.text(), // error message if failed
    processingTimeMs: d.integer(), // time taken to process the request
    createdAt: d
      .timestamp({ withTimezone: true })
      .default(sql`CURRENT_TIMESTAMP`)
      .notNull(),
  }),
  (t) => [
    index("transcript_request_youtube_id_idx").on(t.youtubeId),
    index("transcript_request_created_at_idx").on(t.createdAt),
    index("transcript_request_success_idx").on(t.success),
  ],
);

export const transcriptRequestsRelations = relations(
  transcriptRequests,
  ({ one }) => ({
    // No direct relations needed for this tracking table
  }),
);

// Chapter processing tables for knowledge graph
export const chapters = createTable(
  "chapter",
  (d) => ({
    id: d.integer().primaryKey().generatedByDefaultAsIdentity(),
    videoId: d
      .integer()
      .notNull()
      .references(() => videos.id, { onDelete: "cascade" }),
    chapterIdx: d.integer().notNull(), // YouTube chapter index
    chapterName: d.varchar({ length: 500 }).notNull(),
    chapterSummary: d.text().notNull(), // LLM-generated summary
    startTime: d.integer().notNull(), // start time in seconds
    endTime: d.integer().notNull(), // end time in seconds
    createdAt: d
      .timestamp({ withTimezone: true })
      .default(sql`CURRENT_TIMESTAMP`)
      .notNull(),
    updatedAt: d.timestamp({ withTimezone: true }).$onUpdate(() => new Date()),
  }),
  (t) => [
    index("chapter_video_id_idx").on(t.videoId),
    index("chapter_video_idx_idx").on(t.videoId, t.chapterIdx), // Unique constraint
  ],
);

export const chapterSimilarities = createTable(
  "chapter_similarity",
  (d) => ({
    id: d.integer().primaryKey().generatedByDefaultAsIdentity(),
    sourceChapterId: d
      .integer()
      .notNull()
      .references(() => chapters.id, { onDelete: "cascade" }),
    destChapterId: d
      .integer()
      .notNull()
      .references(() => chapters.id, { onDelete: "cascade" }),
    similarityScore: d.real().notNull(), // Pinecone similarity score
    createdAt: d
      .timestamp({ withTimezone: true })
      .default(sql`CURRENT_TIMESTAMP`)
      .notNull(),
  }),
  (t) => [
    index("chapter_similarity_source_idx").on(t.sourceChapterId),
    index("chapter_similarity_dest_idx").on(t.destChapterId),
    index("chapter_similarity_score_idx").on(t.similarityScore),
    unique("chapter_similarity_unique").on(t.sourceChapterId, t.destChapterId),
  ],
);

// Chapter relations
export const chaptersRelations = relations(chapters, ({ one, many }) => ({
  video: one(videos, {
    fields: [chapters.videoId],
    references: [videos.id],
  }),
  sourceSimilarities: many(chapterSimilarities, {
    relationName: "sourceChapter",
  }),
  destSimilarities: many(chapterSimilarities, { relationName: "destChapter" }),
}));

export const chapterSimilaritiesRelations = relations(
  chapterSimilarities,
  ({ one }) => ({
    sourceChapter: one(chapters, {
      fields: [chapterSimilarities.sourceChapterId],
      references: [chapters.id],
      relationName: "sourceChapter",
    }),
    destChapter: one(chapters, {
      fields: [chapterSimilarities.destChapterId],
      references: [chapters.id],
      relationName: "destChapter",
    }),
  }),
);
