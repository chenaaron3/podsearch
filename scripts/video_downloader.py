#!/usr/bin/env python3
"""
Simple YouTube Video Downloader

Downloads a single YouTube video as MP4 file only.
No thumbnails, subtitles, or metadata files.

Usage:
    from video_downloader import YouTubeDownloader
    
    downloader = YouTubeDownloader("./downloads")
    file_path = downloader.download_video("https://www.youtube.com/watch?v=VIDEO_ID")
    
    if file_path:
        print(f"Downloaded: {file_path}")
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import yt_dlp


class YouTubeDownloader:
    def __init__(self, output_dir: str = "./downloads"):
        """
        Initialize YouTube downloader.
        
        Args:
            output_dir: Directory to save downloaded videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.downloaded_file_path = None
        
        # Simple options - MP4 only, no extra files
        self.base_options = {
            'format': 'best[ext=mp4]/best',  # Prefer MP4, fallback to best
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'writesubtitles': False,
            'writeautomaticsub': False,
            'writedescription': False,
            'writeinfojson': False,
            'writethumbnail': False,
            'embedsubs': False,
            'embedthumbnail': False,
            'addmetadata': False,
            'ignoreerrors': False,
            'no_warnings': True,
        }

    def progress_hook(self, d: Dict[str, Any]) -> None:
        """Progress callback for yt-dlp."""
        if d['status'] == 'downloading':
            try:
                percent = d.get('_percent_str', 'N/A')
                speed = d.get('_speed_str', 'N/A')
                filename = d.get('filename', 'Unknown')
                display_name = Path(filename).name[:50] + "..." if len(Path(filename).name) > 50 else Path(filename).name
                print(f"\r📥 Downloading: {display_name} | {percent} | {speed}", end='', flush=True)
            except Exception:
                print(f"\r📥 Downloading... {d.get('_percent_str', '')}", end='', flush=True)
                
        elif d['status'] == 'finished':
            filename = d.get('filename', 'Unknown')
            self.downloaded_file_path = filename
            print(f"\n✅ Downloaded: {Path(filename).name}")
            
        elif d['status'] == 'error':
            print(f"\n❌ Error downloading: {d.get('filename', 'Unknown')}")

    def download_channel(self,
                        channel_url: str,
                        quality: str = "best",
                        min_duration_minutes: int = 30,
                        max_videos: Optional[int] = None) -> List[str]:
        """
        Download all videos from a YouTube channel with filtering.
        
        Args:
            channel_url: YouTube channel URL (supports @handle, /c/, /channel/, /user/ formats)
            quality: Video quality ('best', '720p', '480p', etc.)
            min_duration_minutes: Minimum video duration in minutes
            max_videos: Maximum number of videos to download (None = all)
            
        Returns:
            List of downloaded file paths
        """
        # Store downloaded files as instance variable to avoid scope issues
        self.channel_downloaded_files = []
        
        try:
            # Use channel URL directly but force single playlist mode
            print(f"🚀 Starting channel download: {channel_url}")
            
            # Prepare download options for channel
            options = self.base_options.copy()
            
            # Set quality format (always MP4)
            if quality == "best":
                options['format'] = 'best[ext=mp4]/best'
            elif quality.endswith('p'):
                height = quality[:-1]
                options['format'] = f'best[height<={height}][ext=mp4]/best[height<={height}]'
            else:
                options['format'] = 'best[ext=mp4]/best'
            
            # Force single playlist mode by extracting flat playlist
            options['extract_flat'] = False  # We need full info for duration filtering
            options['playliststart'] = 1
            
            # Duration filtering - using a more explicit filter function
            min_duration_seconds = min_duration_minutes * 60
            
            def duration_filter(info_dict):
                duration = info_dict.get('duration')
                if duration is None:
                    print(f"⚠️  Duration unknown for {info_dict.get('title', 'Unknown')}, skipping")
                    return "Duration unknown"
                if duration < min_duration_seconds:
                    print(f"⏭️  Skipping {info_dict.get('title', 'Unknown')[:50]}... ({duration}s < {min_duration_seconds}s)")
                    return "Too short"
                print(f"✅ Keeping {info_dict.get('title', 'Unknown')[:50]}... ({duration}s)")
                return None  # None means keep the video
            
            options['match_filter'] = duration_filter
            
            # Limit number of videos if specified
            if max_videos:
                options['playlistend'] = max_videos
            
            # Force quiet mode initially to reduce spam
            options['quiet'] = True
            options['no_warnings'] = True
            
            # Simpler progress hook
            options['progress_hooks'] = [self.channel_progress_hook]
            
            # Download from channel
            with yt_dlp.YoutubeDL(options) as ydl:
                print(f"📺 Filtering for videos >= {min_duration_minutes} minutes ({min_duration_seconds}s)")
                if max_videos:
                    print(f"📊 Maximum videos to check: {max_videos}")
                print(f"🔄 Processing channel...")
                
                ydl.download([channel_url])
                
                print(f"\n🎉 Channel download completed!")
                print(f"📁 Downloaded {len(self.channel_downloaded_files)} videos")
                
                return self.channel_downloaded_files.copy()
                
        except Exception as e:
            print(f"❌ Channel download failed: {e}")
            import traceback
            print(f"🔍 Debug info: {traceback.format_exc()}")
            return self.channel_downloaded_files.copy() if hasattr(self, 'channel_downloaded_files') else []

    def channel_progress_hook(self, d: Dict[str, Any]) -> None:
        """Progress callback for channel downloads."""
        if d['status'] == 'finished':
            filename = d.get('filename', 'Unknown')
            if hasattr(self, 'channel_downloaded_files'):
                self.channel_downloaded_files.append(filename)
            print(f"\n✅ Downloaded: {Path(filename).name}")
        elif d['status'] == 'downloading':
            try:
                percent = d.get('_percent_str', 'N/A')
                speed = d.get('_speed_str', 'N/A')
                filename = d.get('filename', 'Unknown')
                display_name = Path(filename).name[:50] + "..." if len(Path(filename).name) > 50 else Path(filename).name
                print(f"\r📥 Downloading: {display_name} | {percent} | {speed}", end='', flush=True)
            except Exception:
                print(f"\r📥 Downloading... {d.get('_percent_str', '')}", end='', flush=True)
        elif d['status'] == 'error':
            print(f"\n❌ Error downloading: {d.get('filename', 'Unknown')}")

    def download_video(self, 
                      url: str, 
                      quality: str = "best",
                      custom_name: Optional[str] = None) -> Optional[str]:
        """
        Download a single video as MP4.
        
        Args:
            url: YouTube video URL
            quality: Video quality ('best', '720p', '480p', etc.)
            custom_name: Custom filename (without extension)
            
        Returns:
            Local file path of downloaded MP4 if successful, None otherwise
        """
        try:
            # Reset the downloaded file path
            self.downloaded_file_path = None
            
            # Prepare download options
            options = self.base_options.copy()
            options['progress_hooks'] = [self.progress_hook]
            
            # Set quality format (always MP4)
            if quality == "best":
                options['format'] = 'best[ext=mp4]/best'
            elif quality.endswith('p'):
                # Specific resolution like '720p'
                height = quality[:-1]
                options['format'] = f'best[height<={height}][ext=mp4]/best[height<={height}]'
            else:
                options['format'] = 'best[ext=mp4]/best'
            
            # Custom filename
            if custom_name:
                safe_name = self.sanitize_filename(custom_name)
                options['outtmpl'] = str(self.output_dir / f'{safe_name}.%(ext)s')
            
            # Download
            with yt_dlp.YoutubeDL(options) as ydl:
                print(f"🚀 Starting download: {url}")
                ydl.download([url])
                
                # Return the downloaded file path if successful
                if self.downloaded_file_path:
                    print("✅ Download completed successfully!")
                    return self.downloaded_file_path
                else:
                    print("⚠️ Download completed but file path not captured")
                    return None
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file system usage."""
        # Remove or replace problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\s+', '_', filename)  # Replace spaces with underscores
        filename = filename.strip('.')  # Remove leading/trailing dots
        return filename[:200]  # Limit length

    @staticmethod
    def is_channel_url(url: str) -> bool:
        """Check if URL is a channel URL."""
        channel_patterns = [
            r'youtube\.com/@[^/\s]+',
            r'youtube\.com/c/[^/\s]+',
            r'youtube\.com/channel/[^/\s]+',
            r'youtube\.com/user/[^/\s]+'
        ]
        return any(re.search(pattern, url) for pattern in channel_patterns)

    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get video information without downloading.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video information dictionary or None if error
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return info
        except Exception as e:
            print(f"❌ Error getting video info: {e}")
            return None


def main():
    """Simple command line interface."""
    # if len(sys.argv) < 2:
    #     print("Usage: python video_downloader.py <youtube_url> [quality] [options]")
    #     print("\nExamples:")
    #     print("Single video:")
    #     print('  python video_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID"')
    #     print('  python video_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID" "720p"')
    #     print('  python video_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID" "best" "my_video"')
    #     print("\nChannel downloads:")
    #     print('  python video_downloader.py "https://www.youtube.com/@channelname"')
    #     print('  python video_downloader.py "https://www.youtube.com/@channelname" "720p"')
    #     print('  python video_downloader.py "https://www.youtube.com/@channelname" "best" "30" "10"')
    #     print("     (30 = min duration in minutes, 10 = max videos)")
    #     return
    
    url = "https://www.youtube.com/@TheDiaryOfACEO"
    quality = sys.argv[2] if len(sys.argv) > 2 else "best"
    
    downloader = YouTubeDownloader()
    
    # Detect if it's a channel URL or single video URL
    if downloader.is_channel_url(url):
        # Channel download
        min_duration = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 30
        max_videos = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else None
        
        print(f"🔍 Detected channel URL")
        file_paths = downloader.download_channel(url, quality, min_duration, max_videos)
        
        if file_paths:
            print(f"\n📁 Downloaded {len(file_paths)} files:")
            for i, path in enumerate(file_paths, 1):
                print(f"  {i}. {path}")
        else:
            print("❌ No videos downloaded")
    else:
        # Single video download
        custom_name = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].isdigit() else None
        
        print(f"🔍 Detected single video URL")
        file_path = downloader.download_video(url, quality, custom_name)
        
        if file_path:
            print(f"📁 Downloaded to: {file_path}")
        else:
            print("❌ Download failed")


if __name__ == "__main__":
    main()
