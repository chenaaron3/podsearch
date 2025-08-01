import json
import os
from pathlib import Path
from typing import List, Dict, Optional, TypedDict
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

from database import DatabaseManager, VideoStatus
from process_video import VideoProcessor

load_dotenv()

class ChapterData(TypedDict):
    title: str
    start: float
    end: float
class ChapterResults(TypedDict):
    chapters: List[ChapterData]

class HookPattern(TypedDict):
    sentence_index: int
    primary_pattern: str
    explanation: str
    confidence: int

class HookResults(TypedDict):
    hooks: List[HookPattern]

def get_youtube_chapters(video_id: str) -> ChapterResults:
    """
    Get YouTube video chapters for a given video ID using YouTube Data API v3.
    
    Args:
        video_id (str): YouTube video ID (e.g., "dQw4w9WgXcQ")
        
    Returns:
        Dict with chapters list containing title, start, and end times
        Format: {"chapters": [{"title": str, "start": int, "end": int}, ...]}
    """
    
    # Create processed directory if it doesn't exist
    processed_dir = Path("processed/chapters")
    processed_dir.mkdir(exist_ok=True)
    
    # Cache file path
    cache_file = processed_dir / f"{video_id}_chapters.json"
    
    # Check if we have cached data
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if cached_data.get("chapters"):
                    return cached_data
        except (json.JSONDecodeError, IOError):
            pass  # Continue to fetch fresh data if cache is corrupted
    
    try:
        # Use yt-dlp to get chapters directly from YouTube's internal data
        import subprocess
        
        try:
            # Use yt-dlp to extract only metadata (much faster)
            cmd = [
                "yt-dlp",
                "--dump-json",
                "--no-playlist",
                "--no-warnings",
                f"https://www.youtube.com/watch?v={video_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"Error fetching video info for {video_id}: {result.stderr}")
                return {"chapters": []}
            
            # Parse the JSON output
            video_data = json.loads(result.stdout)
            
            # Extract chapters from yt-dlp output
            chapters = []
            if "chapters" in video_data and video_data["chapters"]:
                for chapter in video_data["chapters"]:
                    chapters.append({
                        "title": chapter.get("title", ""),
                        "start": int(chapter.get("start_time", 0)),
                        "end": int(chapter.get("end_time", 0))
                    })
            
            print(f"Found {len(chapters)} chapters for video {video_id}")
            
        except subprocess.TimeoutExpired:
            print(f"Timeout fetching video info for {video_id}")
            return {"chapters": []}
        except json.JSONDecodeError:
            print(f"Error parsing video data for {video_id}")
            return {"chapters": []}
        except Exception as e:
            print(f"Error using yt-dlp for {video_id}: {e}")
            return {"chapters": []}
        
        # Create result
        result_data = {"chapters": chapters}
        
        # Cache the result
        try:
            with open(cache_file, 'w') as f:
                json.dump(result_data, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not cache chapters for {video_id}: {e}")
        
        return result_data
        
    except HttpError as e:
        print(f"YouTube API error for {video_id}: {e}")
        return {"chapters": []}
    except Exception as e:
        print(f"Unexpected error fetching chapters for {video_id}: {e}")
        return {"chapters": []}
    
def get_hook_from_transcript(video_id: str, chapter: ChapterData, sentences: List[str]) -> Optional[HookResults]:
    """
    Generate hooks from transcript sentences using OpenAI API.
    
    Args:
        video_id: YouTube video ID
        chapter: Chapter data with start time
        sentences: List of transcript sentences for the chapter
        
    Returns:
        HookResults with list of hook patterns or None if failed
    """
    from openai import OpenAI
    import hashlib
    
    # Create cache directory if it doesn't exist
    cache_dir = Path("processed/hooks")
    cache_dir.mkdir(exist_ok=True)
    
    # Create cache key based on video_id and chapter start time
    cache_key = f"{video_id}_{int(chapter['start'])}"
    cache_file = cache_dir / f"{cache_key}_hook.json"
    
    # Check if we have cached data
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if cached_data.get("hook"):
                    print(f"📄 Using cached hook for {video_id} chapter at {chapter['start']}s")
                    return cached_data["hook"]
        except (json.JSONDecodeError, IOError):
            pass  # Continue to generate fresh data if cache is corrupted
    
    # Read the prompt from hook.txt
    prompt_file = Path("scripts/prompts/hook.txt")
    if not prompt_file.exists():
        print(f"❌ Hook prompt file not found: {prompt_file}")
        return None
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except IOError as e:
        print(f"❌ Error reading hook prompt: {e}")
        return None
    
    # Replace the {{ transcript }} placeholder with the sentences
    transcript_text = " ".join(sentences)
    prompt = prompt_template.replace("{{ transcript }}", transcript_text)
    
    # Use the OpenAI API to identify hooks
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable is required")
            return None
        
        client = OpenAI(api_key=api_key)
        
        print(f"🤖 Generating hook for {video_id} chapter at {chapter['start']}s...")
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # Using full model for schema validation support
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at identifying compelling hooks and opening lines from video transcripts."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=500,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        
        hook = response.choices[0].message.content.strip()
        
        if not hook:
            print(f"⚠️  No hook generated for {video_id} chapter at {chapter['start']}s")
            return None
        
        # Save the hook to cache
        try:
            cache_data = {
                "video_id": video_id,
                "chapter_start": chapter['start'],
                "chapter_title": chapter['title'],
                "hook": hook,
                "generated_at": datetime.now().isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            print(f"💾 Hook cached for {video_id} chapter at {chapter['start']}s")
            
        except IOError as e:
            print(f"⚠️  Warning: Could not cache hook for {video_id}: {e}")
        
        return hook
        
    except Exception as e:
        print(f"❌ Error generating hook for {video_id}: {e}")
        return None

def get_hooks_from_playlist(playlist_id: str) -> Dict[str, List[Dict[str, any]]]:
    db_manager = DatabaseManager()
    video_processor = VideoProcessor(skip_initialization=True)
    # get videos for playist
    videos = db_manager.get_videos_by_status(VideoStatus.FINISHED, playlist_id)
    # for each video, get the chapters
    for video in videos:
        print(f"Processing video {video.youtube_id} {video.id}")
        chapters = get_youtube_chapters(video.youtube_id)
        # using the chapters and the transcript, find the first sentence of each chapter
        transcript = video_processor.extract_transcript(video)
        if transcript is None:
            print(f"No transcript found for video {video.youtube_id}")
            continue
        for chapter in chapters['chapters']:
            print()
            print("Processing chapter", chapter)
            chapter_sentences = []
            # find sentences that fall within the chapter timeframe
            for sentence in transcript["segments"]:
                if sentence["start"] >= chapter["start"] and sentence["end"] <= chapter["end"]:
                    chapter_sentences.append(sentence["text"])
            
            if chapter_sentences:
                print(f"Found {len(chapter_sentences)} sentences for chapter")
                # Generate hook from the chapter sentences
                hook = get_hook_from_transcript(video.youtube_id, chapter, chapter_sentences)
                if hook:
                    print(f"🎣 Generated hook: {hook}")
                else:
                    print("❌ Failed to generate hook")
            else:
                print("⚠️  No sentences found for this chapter")

# Example usage
if __name__ == "__main__":
    get_hooks_from_playlist(5)
