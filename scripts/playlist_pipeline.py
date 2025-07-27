#!/usr/bin/env python3
"""
Complete Video Processing Pipeline for YouTube Playlists and Channels

Orchestrates the entire workflow:
1. Fetch playlist/channel videos from YouTube (only ≥30 minutes)
2. Download videos (pending -> downloaded)  
3. Process videos with Whisper (downloaded -> processed)
4. Generate embeddings (processed -> embedded)
5. Complete processing (embedded -> finished)

Usage:
    python playlist_pipeline.py <playlist_or_channel_url>
"""

import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from database import DatabaseManager, Video, VideoStatus
from playlist_fetcher import PlaylistFetcher
from video_downloader import YouTubeDownloader
from process_video import VideoProcessor

class PlaylistPipeline:
    def __init__(self, 
                 downloads_dir: str = "./downloads",
                 processed_dir: str = "./processed"):
        """Initialize the complete pipeline."""
        self.db_manager = DatabaseManager()
        
        self.playlist_fetcher = PlaylistFetcher(self.db_manager)
        self.downloader = YouTubeDownloader(downloads_dir)
        self.processor = VideoProcessor(
            downloads_dir=downloads_dir,
            output_dir=processed_dir
        )
        
        self.downloads_dir = Path(downloads_dir)
        self.processed_dir = Path(processed_dir)
        
        # Create directories
        self.downloads_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        print(f"🔧 Pipeline initialized:")
        print(f"   Downloads: {self.downloads_dir}")
        print(f"   Processed: {self.processed_dir}")
        print(f"   Minimum duration: {self.playlist_fetcher.min_duration_seconds/60:.0f} minutes")
    
    def sync_playlist(self, url: str) -> tuple:
        """Sync playlist and return (playlist, new_videos_count)."""
        print(f"\n🎵 Step 1: Syncing playlist/channel...")
        return self.playlist_fetcher.sync_playlist(url)
    
    def process_single_video(self, video: Video) -> bool:
        """
        Process a single video through all stages.
        Returns True if successful, False if failed.
        """
        try:
            # Stage 1: Download video
            if not self._download_video(video):
                return False
            
            # Stage 2: Process video (transcript + embeddings)  
            if not self._process_video(video):
                return False
            
            print(f"🎉 Video processing complete: {video.title[:50]}...")
            
            return True
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.db_manager.update_video_status(video.id, VideoStatus.FAILED, error_msg)
            print(f"❌ Video failed: {error_msg}")
            return False
    
    def _download_video(self, video: Video) -> bool:
        """Download video and update status."""
        try:
            print(f"  📥 Downloading...")
            
            # Check if already downloaded
            if video.local_file_path:
                local_path = Path(video.local_file_path)
                if local_path.exists():
                    print(f"  ✅ Already downloaded: {local_path.name}")
                    return True
            
            # Download video
            local_file_path = self.downloader.download_video(video.url)
            
            if local_file_path and Path(local_file_path).exists():
                abs_path = str(Path(local_file_path).resolve())
                self.db_manager.update_video_status(
                    video.id, 
                    VideoStatus.DOWNLOADED,
                    local_file_path=abs_path
                )
                print(f"  ✅ Downloaded: {Path(abs_path).name}")
                return True
            else:
                self.db_manager.update_video_status(video.id, VideoStatus.FAILED, "Download failed - no file created")
                return False
                
        except Exception as e:
            error_msg = f"Download error: {str(e)}"
            self.db_manager.update_video_status(video.id, VideoStatus.FAILED, error_msg)
            print(f"  ❌ Download failed: {error_msg}")
            return False
    
    def _process_video(self, video: Video) -> bool:
        """Process video (transcript + embeddings) and update status.""" 
        try:
            print(f"  🔄 Processing (transcript + embeddings)...")
            
            # Process video using VideoProcessor with video ID
            # The processor will fetch all metadata from the database
            success = self.processor.process_single_video(
                video.id,
                force_reprocess=False
            )
            
            if success:
                print(f"  ✅ Processing complete")
                return True
            else:
                self.db_manager.update_video_status(video.id, VideoStatus.FAILED, "Video processing failed")
                return False
                
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            self.db_manager.update_video_status(video.id, VideoStatus.FAILED, error_msg)
            print(f"  ❌ Processing failed: {error_msg}")
            return False
    
    def show_playlist_status(self, playlist_id: int):
        """Show current status of all videos in playlist."""
        print(f"\n📊 Playlist Status Summary:")
        print("=" * 50)
        
        status_counts = {}
        with self.db_manager.get_session() as session:
            from sqlalchemy import func
            results = session.query(
                Video.status, 
                func.count(Video.id)
            ).filter(
                Video.playlist_id == playlist_id
            ).group_by(Video.status).all()
            
            for status, count in results:
                status_counts[status] = count
        
        total = sum(status_counts.values())
        
        # Display counts for each status
        for status in VideoStatus.all_values():
            count = status_counts.get(status, 0)
            percentage = (count / total * 100) if total > 0 else 0
            status_name = status.capitalize()
            print(f"  {status_name:>10}: {count:>3} ({percentage:>5.1f}%)")
        
        print(f"  {'Total':>10}: {total:>3}")
        print("=" * 50)
    
    def process_by_status(self, playlist_id: int):
        """Resume processing from where it left off."""
        print(f"\n🔄 Resuming processing...")
        
        # Process videos in each stage
        stages_to_check = [
            VideoStatus.EMBEDDED,
            VideoStatus.FAILED,
            VideoStatus.DOWNLOADED,
            VideoStatus.PENDING,
        ]
        
        for status in stages_to_check:
            videos = self.db_manager.get_videos_by_status(status, playlist_id)
            if videos:
                print(f"\n📹 Found {len(videos)} videos in '{status}' status")
                for video in videos:
                    print(f"  Resuming: {video.title[:50]}...")
                    self.process_single_video(video)
        
        print("\n✅ Resume complete")

def main():
    """Main pipeline entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Playlist/Channel Processing Pipeline",
        epilog="""
Examples:
  python playlist_pipeline.py --url "https://www.youtube.com/playlist?list=PLExample"
  python playlist_pipeline.py --playlist-id 123
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", help="YouTube playlist or channel URL")
    parser.add_argument("--playlist-id", type=int, help="Filter by playlist ID when batch processing")
    parser.add_argument("--downloads-dir", default="./downloads", help="Downloads directory")
    parser.add_argument("--processed-dir", default="./processed", help="Processed output directory")
    
    args = parser.parse_args()
    
    try:
        print("🎬 YouTube Playlist/Channel Processing Pipeline")
        print("=" * 60)
        print(f"📺 Processing: {args.url}")
        print(f"⏱️ Minimum duration: 30 minutes")
        print()
        
        # Initialize pipeline
        pipeline = PlaylistPipeline(
            downloads_dir=args.downloads_dir,
            processed_dir=args.processed_dir
        )
        
        if args.url:
            # Sync playlist (always do this to get latest videos)
            playlist, new_count = pipeline.sync_playlist(args.url)
        elif args.playlist_id:
            playlist = pipeline.db_manager.get_playlist_by_id(args.playlist_id)
        
        # Show current status
        pipeline.show_playlist_status(playlist.id)
        
        # Process all unfinished videos
        pipeline.process_by_status(playlist.id)
        
        # Show final status
        print(f"\n🏁 Pipeline Complete!")
        pipeline.show_playlist_status(playlist.id)
        
        # Show some stats
        with pipeline.db_manager.get_session() as session:
            from sqlalchemy import func
            finished_count = session.query(func.count(Video.id)).filter(
                Video.playlist_id == playlist.id,
                Video.status == VideoStatus.FINISHED
            ).scalar()
            
            failed_count = session.query(func.count(Video.id)).filter(
                Video.playlist_id == playlist.id,
                Video.status == VideoStatus.FAILED
            ).scalar()
            
            if finished_count > 0:
                print(f"\n🎉 Successfully processed {finished_count} videos!")
                print(f"🔍 You can now search these videos using:")
                print(f"   python search_segments.py \"your query\"")
            
            if failed_count > 0:
                print(f"\n⚠️ {failed_count} videos failed processing")
                print(f"💡 Check error messages in database for details")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        print(f"🔍 Debug info: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 