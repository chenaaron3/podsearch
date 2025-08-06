#!/usr/bin/env python3
"""
Chapter Processing Pipeline for Knowledge Graph Construction

Processes YouTube chapters from finished videos:
1. Extract chapters from YouTube API
2. Generate summaries using LLM
3. Create embeddings and store in Pinecone
4. Find similar chapters across videos
5. Store relationships in database

Usage:
    python chapter_pipeline.py --playlist-id 123
    python chapter_pipeline.py --all-finished
"""

import sys
import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from pinecone import Pinecone, ServerlessSpec

from database import DatabaseManager, Video, Chapter, ChapterSimilarity, TranscriptData, ChapterSimilarityData
from hook_finder import get_youtube_chapters, ChapterResults, ChapterData
from process_video import VideoProcessor

class ChapterProcessor:
    def __init__(self):
        """Initialize the chapter processing pipeline."""
        self.db_manager = DatabaseManager()
        self.video_processor = VideoProcessor()
        
        # Initialize OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Get or create Pinecone index for chapters
        self.index_name = "chapter-summaries"
        self._setup_pinecone_index()
        
        # Load summarization prompt
        self.summary_prompt = self._load_summary_prompt()
        
        print(f"🔧 Chapter processor initialized:")
        print(f"   Pinecone index: {self.index_name}")
        print(f"   OpenAI model: gpt-4.1-mini")
    
    def _setup_pinecone_index(self):
        """Set up Pinecone index for chapter summaries."""
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                print(f"📊 Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=3072,  # OpenAI text-embedding-3-large dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"✅ Pinecone index created")
            else:
                print(f"✅ Pinecone index already exists: {self.index_name}")
                
        except Exception as e:
            print(f"❌ Error setting up Pinecone index: {e}")
            raise
    
    def _load_summary_prompt(self) -> str:
        """Load the summarization prompt."""
        prompt_path = Path(__file__).parent / "prompts" / "summarize.txt"
        try:
            with open(prompt_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"❌ Summary prompt not found at: {prompt_path}")
            raise
    
    def process_video_chapters(self, video: Video) -> bool:
        """
        Process chapters for a single video.
        Returns True if successful, False if failed.
        """
        try:
            print(f"\n📹 Processing chapters for: {video.title[:50]}...")
            
            # Check if chapters already exist for this video
            existing_chapters = self.db_manager.get_chapters_by_video_id(video.id)
            if existing_chapters:
                print(f"  ✅ Chapters already exist for video {video.youtube_id} ({len(existing_chapters)} chapters)")
                return True  # Skip processing, chapters already exist
            
            # Step 1: Get chapters from YouTube
            chapters = self._get_video_chapters(video.youtube_id)
            if not chapters:
                print(f"  ⚠️ No chapters found for video {video.youtube_id}")
                return True  # Not an error, just no chapters
            
            # Step 2: Get transcript data
            transcript_data = self._get_video_transcript(video)
            if not transcript_data:
                print(f"  ❌ No transcript found for video {video.youtube_id}")
                return False
            
            # Step 3: Process each chapter
            processed_chapters = []
            for chapter_idx, chapter in enumerate(chapters['chapters']):
                print(f"  📖 Processing chapter {chapter_idx + 1}: {chapter['title'][:30]}...")
                
                # Extract chapter text from transcript
                chapter_text = self._extract_chapter_text(transcript_data, chapter)
                if not chapter_text:
                    print(f"    ⚠️ No transcript text found for chapter")
                    continue
                
                # Generate summary
                summary = self._generate_chapter_summary(chapter_text)
                if not summary:
                    print(f"    ❌ Failed to generate summary")
                    continue

                # Save chapter to database
                chapter_obj = self.db_manager.save_chapter(
                    video_id=video.id,
                    chapter_idx=chapter_idx,
                    chapter_name=chapter['title'],
                    chapter_summary=summary,
                    start_time=int(chapter['start']),
                    end_time=int(chapter['end'])
                )
                
                if chapter_obj:
                    # Generate and store embedding for the chapter
                    self._process_chapter_embedding(chapter_obj)
                    
                    processed_chapters.append(chapter_obj)
                    print(f"    ✅ Saved chapter: {summary[:50]}...")
            
            print(f"  🎉 Processed {len(processed_chapters)} chapters")
            return True
            
        except Exception as e:
            print(f"❌ Error processing chapters for video {video.youtube_id}: {e}")
            return False
    
    def _get_video_chapters(self, youtube_id: str) -> Optional[ChapterResults]:
        """Get chapters from YouTube API."""
        try:
            return get_youtube_chapters(youtube_id)
        except Exception as e:
            print(f"  ❌ Error getting chapters: {e}")
            return None
    
    def _get_video_transcript(self, video: Video) -> Optional[TranscriptData]:
        """Get transcript data for a video."""
        return self.video_processor.extract_transcript(video, force_reprocess=False)
    
    def _extract_chapter_text(self, transcript_data: TranscriptData, chapter: ChapterData) -> str:
        """Extract text from transcript that falls within chapter timeframe."""
        chapter_sentences = []
        
        for sentence in transcript_data["segments"]:
            if sentence["start"] >= chapter["start"] and sentence["end"] <= chapter["end"]:
                chapter_sentences.append(sentence["text"])
        
        return " ".join(chapter_sentences)
    
    def _generate_chapter_summary(self, chapter_text: str) -> Optional[str]:
        """Generate a summary for chapter text using OpenAI."""
        try:
            # Prepare prompt
            prompt = self.summary_prompt.replace("{{transcript}}", chapter_text)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts the main concept from podcast chapters. Return your response as a JSON object with a 'summary' field."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            
            # Parse JSON response
            response_content = response.choices[0].message.content.strip()
            response_data = json.loads(response_content)
            
            summary = response_data.get("summary", "")
            return summary if summary else None
            
        except json.JSONDecodeError as e:
            print(f"    ❌ Failed to parse JSON response: {e}")
            return None
        except Exception as e:
            print(f"    ❌ OpenAI API error: {e}")
            return None
    
    def process_similarities(self):
        """Process similarities for chapters, batching by videos to manage memory."""
        print(f"\n🔗 Processing chapter similarities...")
        

        """Process similarities for all videos, one video at a time."""
        print(f"  📹 Processing similarities for all videos")
        
        # Get all videos that have chapters
        videos_with_chapters = self.db_manager.get_videos_with_chapters()
        
        if not videos_with_chapters:
            print("  ⚠️ No videos with chapters found")
            return
        
        print(f"  📊 Found {len(videos_with_chapters)} videos with chapters")
        
        # Process each video
        for i, video in enumerate(videos_with_chapters, 1):
            print(f"\n  📹 [{i}/{len(videos_with_chapters)}] Processing video: {video.title[:50]}...")
            self._process_similarities_for_video(video.id)
    
    def _process_similarities_for_video(self, video_id: int):
        """Process similarities for all chapters in a single video."""
        # Get chapters for this video
        chapters = self.db_manager.get_chapters_by_video_id(video_id)
        
        if not chapters:
            print(f"    ⚠️ No chapters found for video {video_id}")
            return
        
        print(f"    📊 Processing {len(chapters)} chapters for video {video_id}")
        
        # Process each chapter in this video
        for i, chapter in enumerate(chapters, 1):
            print(f"    🔗 [{i}/{len(chapters)}] Processing chapter: {chapter.chapter_name[:30]}...")
            
            # Get the embedding for similarity search
            embedding = self._process_chapter_embedding(chapter)
            if not embedding:
                if not embedding:
                    print(f"      ❌ No embedding found for chapter")
                continue
            
            # Find similar chapters
            similar_chapters = self._find_similar_chapters(chapter, embedding)
            
            # Print chapter details for quality evaluation
            print(f"\n      🎯 Source Chapter (Similarity Score: N/A):")
            # self._print_chapter_details(chapter)
            # Prepare similarities for bulk upsert
            similarities_data: List[ChapterSimilarityData] = []
            for similar_chapter, score in similar_chapters:
                print(f"\n      🔗 Similar Chapter (Score: {score:.3f}):")
                # self._print_chapter_details(similar_chapter)
                similarities_data.append(ChapterSimilarityData(
                    source_chapter_id=chapter.id,
                    dest_chapter_id=similar_chapter.id,
                    similarity_score=score
                ))
            
            # Bulk upsert all similarities in one operation
            if similarities_data:
                affected_rows = self.db_manager.bulk_upsert_chapter_similarities(similarities_data)
                print(f"      ✅ Found and saved {len(similar_chapters)} similar chapters ({affected_rows} rows affected)")
            else:
                print(f"      ✅ Found {len(similar_chapters)} similar chapters (none to save)")
    
    def _get_chapter_embedding(self, chapter: Chapter) -> Optional[List[float]]:
        """Get embedding for a chapter summary."""
        try:
            response = self.openai_client.embeddings.create(
                input=chapter.chapter_summary,
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"    ❌ Error getting embedding: {e}")
            return None
    
    def _store_chapter_embedding(self, chapter: Chapter, embedding: List[float]):
        """Store chapter embedding in Pinecone."""
        try:
            index = self.pc.Index(self.index_name)
            
            vector = {
                "id": str(chapter.id),
                "values": embedding,
                "metadata": {
                    "chapter_id": chapter.id,
                    "video_id": chapter.video_id,
                    "chapter_name": chapter.chapter_name,
                    "chapter_summary": chapter.chapter_summary[:500]  # Limit metadata size
                }
            }
            
            index.upsert(vectors=[vector])
            
        except Exception as e:
            print(f"    ❌ Error storing embedding in Pinecone: {e}")
            raise
    
    def _get_stored_chapter_embedding(self, chapter: Chapter) -> Optional[List[float]]:
        """Get stored chapter embedding from Pinecone."""
        try:
            index = self.pc.Index(self.index_name)
            
            # Fetch the vector by ID
            fetch_response = index.fetch(ids=[str(chapter.id)])
            
            if str(chapter.id) in fetch_response.vectors:
                vector = fetch_response.vectors[str(chapter.id)]
                return vector.values
            else:
                return None
                
        except Exception as e:
            print(f"    ❌ Error getting stored embedding: {e}")
            return None
    
    def _process_chapter_embedding(self, chapter: Chapter) ->  Optional[List[float]]:
        """
        Generate and store embedding for a chapter.
        Returns the embedding if successful, None if failed.
        """
        try:
            print(f"    🔗 Generating embedding...")
            
            # Check if embedding already exists
            existing_embedding = self._get_stored_chapter_embedding(chapter)
            if existing_embedding:
                print(f"    ✅ Embedding already exists in Pinecone")
                return existing_embedding
            
            # Generate new embedding
            embedding = self._get_chapter_embedding(chapter)
            if not embedding:
                print(f"    ⚠️ Failed to generate embedding")
                return None
            
            # Store embedding in Pinecone
            self._store_chapter_embedding(chapter, embedding)
            print(f"    ✅ Embedding stored in Pinecone")
            return embedding
            
        except Exception as e:
            print(f"    ❌ Error processing chapter embedding: {e}")
            return None
    
    def _print_chapter_details(self, chapter: Chapter, prefix: str = "      "):
        """Helper function to print chapter transcript and summary."""
        print(f"{prefix}📋 Chapter: {chapter.chapter_name}")
        
        # Convert Chapter to ChapterData for transcript extraction
        chapter_data = {
            "title": chapter.chapter_name,
            "start": chapter.start_time,
            "end": chapter.end_time
        }
        
        # Get video and transcript
        video = self.db_manager.get_video_by_id(chapter.video_id)
        if video:
            transcript_data = self._get_video_transcript(video)
            if transcript_data:
                chapter_text = self._extract_chapter_text(transcript_data, chapter_data)
                print(f"{prefix}📝 Transcript: {chapter_text}...")
        
        print(f"{prefix}📄 Summary: {chapter.chapter_summary}")
        print(f"{prefix}📹 Video ID: {chapter.video_id}")
    
    def _find_similar_chapters(self, source_chapter: Chapter, embedding: List[float], top_k: int = 5) -> List[tuple]:
        """Find similar chapters using Pinecone."""
        try:
            # Query Pinecone
            index = self.pc.Index(self.index_name)
            
            # Query for similar chapters (excluding same video)
            results = index.query(
                vector=embedding,
                top_k=top_k,  # Get more results to filter out same video
                include_metadata=True,
                filter={
                    "video_id": {"$ne": source_chapter.video_id}
                }
            )
            
            # Convert results to chapter objects
            # Fetch all chapter IDs from matches
            match_id_to_score = {}
            chapter_ids = []
            for match in results.matches:
                chapter_id = int(match.id)
                chapter_ids.append(chapter_id)
                match_id_to_score[chapter_id] = match.score

            # Fetch all chapters in one request
            chapters = self.db_manager.get_chapters_by_ids(chapter_ids)
            # Map chapter_id to chapter object for quick lookup
            chapter_id_to_chapter = {chapter.id: chapter for chapter in chapters}

            similar_chapters = []
            for chapter_id in chapter_ids:
                chapter = chapter_id_to_chapter.get(chapter_id)
                if chapter: 
                    similar_chapters.append((chapter, match_id_to_score[chapter_id]))
                    if len(similar_chapters) >= top_k:
                        break
            
            return similar_chapters
            
        except Exception as e:
            print(f"    ❌ Error finding similar chapters: {e}")
            return []
    
    def process_by_playlist(self, playlist_id: int, force_similarities: bool = False, max_workers: int = 4):
        """Process all finished videos in a playlist."""
        print(f"\n🎵 Processing chapters for playlist {playlist_id}")
        
        # Get finished videos without chapters
        videos = self.db_manager.get_finished_videos(playlist_id)
        
        if not videos:
            print("  ✅ All videos already have chapters processed")
            return
        
        print(f"  📹 Found {len(videos)} videos to process")
        print(f"  🔄 Using {max_workers} parallel workers")
        
        # Process videos in parallel
        self._process_videos_parallel(videos, max_workers, "playlist")
        
        # Process similarities for all videos in playlist
        self.process_similarities()
    
    def process_all_finished(self, force_similarities: bool = False, max_workers: int = 4):
        """Process all finished videos across all playlists."""
        print(f"\n🌍 Processing all finished videos")
        
        # Get finished videos without chapters
        videos = self.db_manager.get_finished_videos()
        
        if not videos:
            print("  ✅ All videos already have chapters processed")
            return
        
        print(f"  📹 Found {len(videos)} videos to process")
        print(f"  🔄 Using {max_workers} parallel workers")
        
        # Process videos in parallel
        # self._process_videos_parallel(videos, max_workers, "all")
        
        # Process similarities for all videos
        self.process_similarities()
    
    def _process_videos_parallel(self, videos: List[Video], max_workers: int, context: str):
        """Process multiple videos in parallel using ThreadPoolExecutor."""
        print(f"  🚀 Starting parallel processing with {max_workers} workers...")
        
        results = {
            "total": len(videos),
            "success": 0,
            "failed": 0,
            "errors": []
        }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all video processing tasks
            future_to_video = {
                executor.submit(self._process_single_video, video, i, len(videos)): video 
                for i, video in enumerate(videos, 1)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_video):
                video = future_to_video[future]
                try:
                    success = future.result()
                    if success:
                        results["success"] += 1
                    else:
                        results["failed"] += 1
                except Exception as e:
                    results["failed"] += 1
                    error_msg = f"Error processing video {video.title}: {e}"
                    results["errors"].append(error_msg)
                    print(f"    ❌ {error_msg}")
        
        # Print summary
        print(f"\n📊 {context.capitalize()} processing complete:")
        print(f"   ✅ Success: {results['success']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   📈 Success rate: {results['success']}/{results['total']} ({results['success']/results['total']*100:.1f}%)")
        
        if results["errors"]:
            print(f"\n⚠️ Errors encountered:")
            for error in results["errors"][:5]:  # Show first 5 errors
                print(f"   • {error}")
            if len(results["errors"]) > 5:
                print(f"   ... and {len(results['errors']) - 5} more errors")
    
    def _process_single_video(self, video: Video, index: int, total: int) -> bool:
        """Process a single video (wrapper for parallel execution)."""
        try:
            print(f"    [{index}/{total}] Processing: {video.title[:50]}...")
            success = self.process_video_chapters(video)
            if success:
                print(f"    [{index}/{total}] ✅ Completed: {video.title[:50]}")
            else:
                print(f"    [{index}/{total}] ❌ Failed: {video.title[:50]}")
            return success
        except Exception as e:
            print(f"    [{index}/{total}] ❌ Exception: {video.title[:50]} - {e}")
            return False

def main():
    """Main pipeline entry point."""
    parser = argparse.ArgumentParser(
        description="Chapter Processing Pipeline for Knowledge Graph",
        epilog="""
Examples:
  python chapter_pipeline.py --playlist-id 123
  python chapter_pipeline.py --force-similarities
  python chapter_pipeline.py --max-workers 8
  python chapter_pipeline.py --playlist-id 123 --max-workers 6
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--playlist-id", type=int, help="Process videos in specific playlist")
    parser.add_argument("--force-similarities", action="store_true", help="Force regeneration of similarities even if they exist")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    try:
        print("📚 Chapter Processing Pipeline for Knowledge Graph")
        print("=" * 60)
        
        # Initialize processor
        processor = ChapterProcessor()
        
        if args.playlist_id:
            processor.process_by_playlist(args.playlist_id, force_similarities=args.force_similarities, max_workers=args.max_workers)
        else:
            processor.process_all_finished(force_similarities=args.force_similarities, max_workers=args.max_workers)
        
        print(f"\n🏁 Chapter processing complete!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        print(f"🔍 Debug info: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 