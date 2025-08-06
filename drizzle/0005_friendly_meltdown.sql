DROP INDEX "chapter_similarity_unique_idx";--> statement-breakpoint
ALTER TABLE "podsearch_chapter_similarity" ADD CONSTRAINT "chapter_similarity_unique" UNIQUE("sourceChapterId","destChapterId");