import { relations, sql } from "drizzle-orm";
import { index, pgTableCreator, primaryKey } from "drizzle-orm/pg-core";

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

export const videosRelations = relations(videos, ({ one }) => ({
  playlist: one(playlists, {
    fields: [videos.playlistId],
    references: [playlists.id],
  }),
  transcript: one(transcripts, {
    fields: [videos.transcriptId],
    references: [transcripts.id],
  }),
}));

export const transcriptsRelations = relations(transcripts, ({ one }) => ({
  video: one(videos, {
    fields: [transcripts.id],
    references: [videos.transcriptId],
  }),
}));
