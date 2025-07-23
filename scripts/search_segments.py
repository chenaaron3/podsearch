#!/usr/bin/env python3
"""
Video Segment Search

Search through processed video segments using semantic similarity.
Uses the Pinecone index created by process_video.py

Usage:
    python search_segments.py "What does the guest say about business?"
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class SegmentSearcher:
    def __init__(self, 
                 pinecone_index_name: str = "video-segments",
                 embedding_type: str = "local",
                 local_model_name: str = "all-MiniLM-L6-v2",
                 transcript_dir: str = "processed/transcripts"):
        """
        Initialize the segment searcher.
        
        Args:
            pinecone_index_name: Name of the Pinecone index to search
            embedding_type: "local" or "openai" for embedding method
            local_model_name: Name of local sentence transformer model
            transcript_dir: Directory containing transcript files
        """
        self.index_name = pinecone_index_name
        self.embedding_type = embedding_type
        self.local_model_name = local_model_name
        self.transcript_dir = Path(transcript_dir)
        
        # Initialize embedding model based on type
        if embedding_type == "local":
            print(f"🔄 Loading local embedding model: {local_model_name}...")
            self.embedding_model = SentenceTransformer(local_model_name)
            self.openai_client = None
            print(f"✅ Local model loaded")
        elif embedding_type == "openai":
            print(f"🔄 Initializing OpenAI client...")
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.embedding_model = None
            print(f"✅ OpenAI client initialized")
        else:
            raise ValueError(f"Invalid embedding_type: {embedding_type}. Use 'local' or 'openai'")
        
        # Initialize Pinecone
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pinecone_client.Index(self.index_name)

    def search_segments(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for video segments similar to the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching segments with metadata
        """
        print(f"🔍 Searching for: '{query}' (using {self.embedding_type} embeddings)")
        
        try:
            # Generate embedding for the query based on type
            if self.embedding_type == "local":
                query_embedding = self.embedding_model.encode([query])[0].tolist()
            elif self.embedding_type == "openai":
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[query]
                )
                query_embedding = response.data[0].embedding
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                result = {
                    "score": match.score,
                    "video_name": match.metadata.get("video_name", "Unknown"),
                    "video_url": match.metadata.get("video_url", ""),
                    "timestamp": match.metadata.get("timestamp_readable", ""),
                    "start_time": match.metadata.get("start_time", 0),
                    "duration": match.metadata.get("duration", 0),
                    "text": match.metadata.get("text", "")
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def fetch_full_text_from_transcript(self, result: Dict[str, Any]) -> str:
        """
        Fetch the full segment text from the original transcript file.
        
        Args:
            result: Search result with metadata
            
        Returns:
            Full segment text from transcript
        """
        try:
            video_name = result.get("video_name", "")
            if not video_name:
                return result.get("text", "")
            
            # Find the transcript file
            transcript_file = self._find_transcript_file(video_name)
            if not transcript_file:
                return result.get("text", "")
            
            # Load and parse transcript
            transcript_data = self._load_transcript_file(transcript_file)
            if not transcript_data:
                return result.get("text", "")
            
            # Find matching segments based on timestamp
            start_time = result.get("start_time", 0)
            duration = result.get("duration", 0)
            matching_segments = self._find_matching_segments(
                transcript_data, start_time, duration
            )
            
            if matching_segments:
                full_text = " ".join([seg.get("text", "") for seg in matching_segments])
                return full_text.strip()
            else:
                # Fallback: return original text if no segments found
                return result.get("text", "")
                
        except Exception as e:
            print(f"⚠️  Error fetching full text for {video_name}: {e}")
            return result.get("text", "")
    
    def _find_transcript_file(self, video_name: str) -> Path:
        """Find the transcript file for a given video name."""
        # Remove .mp4 extension if present
        clean_name = video_name.replace('.mp4', '')
        
        # Generate possible transcript file names
        possible_names = []
        
        # Direct match with _transcript suffix
        possible_names.append(f"{clean_name}_transcript.json")
        
        # Handle spaces converted to underscores
        possible_names.append(f"{clean_name.replace(' ', '_')}_transcript.json")
        
        # Handle unicode characters that might be converted
        ascii_name = clean_name.replace('：', ':').replace('，', ',')
        possible_names.append(f"{ascii_name}_transcript.json")
        possible_names.append(f"{ascii_name.replace(' ', '_')}_transcript.json")
        
        # Try each possible name
        for name in possible_names:
            transcript_file = self.transcript_dir / name
            if transcript_file.exists():
                return transcript_file
        
        # Fallback: search for any file containing the video name (without extension)
        search_terms = [clean_name.lower(), ascii_name.lower()]
        for transcript_file in self.transcript_dir.glob("*_transcript.json"):
            file_stem_lower = transcript_file.stem.lower()
            for term in search_terms:
                if term in file_stem_lower:
                    return transcript_file
        
        print(f"⚠️  Transcript file not found for: {video_name}")
        return None
    
    def _load_transcript_file(self, transcript_file: Path) -> Dict:
        """Load and validate transcript file."""
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
                
            if "segments" not in transcript_data:
                print(f"⚠️  Invalid transcript format in {transcript_file.name}")
                return None
                
            return transcript_data
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Error loading transcript {transcript_file.name}: {e}")
            return None
    
    def _find_matching_segments(self, transcript_data: Dict, start_time: float, duration: float) -> List[Dict]:
        """Find transcript segments that overlap with the given time range."""
        end_time = start_time + duration
        matching_segments = []
        
        # Add some tolerance for timestamp matching (±1 second)
        tolerance = 1.0
        
        for segment in transcript_data.get("segments", []):
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", seg_start + 1)
            
            # Check for overlap with tolerance
            if (seg_start - tolerance <= end_time and seg_end + tolerance >= start_time):
                matching_segments.append(segment)
        
        # Sort segments by start time to maintain order
        matching_segments.sort(key=lambda x: x.get("start", 0))
        
        return matching_segments

    def enrich_results_with_full_text(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich search results with full text from transcript files.
        
        Args:
            results: Original search results
            
        Returns:
            Results with full text added
        """
        print(f"🔍 Fetching full text from transcript files...")
        enriched_results = []
        
        for i, result in enumerate(results, 1):
            print(f"   Fetching full text for result {i}...")
            enriched_result = result.copy()
            
            # Fetch full text from transcript
            full_text = self.fetch_full_text_from_transcript(result)
            enriched_result["full_text"] = full_text
            enriched_result["original_text"] = result.get("text", "")
            enriched_result["has_full_text"] = len(full_text) > len(result.get("text", ""))
            
            enriched_results.append(enriched_result)
        
        return enriched_results

    def display_results(self, results: List[Dict[str, Any]], show_full_text: bool = False):
        """Display search results in a formatted way."""
        if not results:
            print("❌ No results found")
            return
        
        print(f"\n📋 Found {len(results)} matching segments:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n🎬 Result {i} - Score: {result['score']:.3f}")
            print(f"📺 Video: {result['video_name']}")
            print(f"⏰ Time: {result['timestamp']} ({result['duration']:.1f}s)")
            if result['video_url'] != "":
                # Create timestamped YouTube URL
                start_seconds = int(result['start_time'])
                timestamped_url = f"{result['video_url']}&t={start_seconds}s"
                print(f"🔗 Link: {timestamped_url}")
            
            # Display text based on what's available
            if show_full_text and result.get('full_text'):
                full_text = result['full_text']
                original_text = result.get('original_text', result.get('text', ''))
                
                print(f"📝 Full Text ({len(full_text)} chars):")
                print(f"   {full_text}")
                
                if len(full_text) > len(original_text):
                    print(f"💡 Original snippet ({len(original_text)} chars): {original_text[:100]}...")
                
            else:
                text = result.get('text', '')
                print(f"📝 Text: {text}")
                
                if result.get('has_full_text'):
                    print(f"💡 Full text available - use --full-text flag to see complete segment")
            
            print("-" * 80)

def main():
    """Main function for interactive search."""
    if len(sys.argv) < 2:
        print("Usage: python search_segments.py '<search_query>' [--full-text]")
        print("\nOptions:")
        print("  --full-text    Fetch and display complete segment text from transcript files")
        print("\nExamples:")
        print('python search_segments.py "What advice do they give about money?"')
        print('python search_segments.py "business tips" --full-text')
        return
    
    # Parse command line arguments
    args = sys.argv[1:]
    show_full_text = "--full-text" in args
    
    # Remove flags from query
    query_parts = [arg for arg in args if not arg.startswith("--")]
    query = " ".join(query_parts)
    
    # Configuration options (should match what was used during processing)
    embedding_type = os.getenv("EMBEDDING_TYPE", "local").lower()
    local_model = os.getenv("LOCAL_MODEL", "all-MiniLM-L6-v2")
    
    print(f"🔧 Search Configuration:")
    print(f"   Embedding type: {embedding_type}")
    if embedding_type == "local":
        print(f"   Local model: {local_model}")
    print()
    
    # Check required environment variables based on embedding type
    required_env_vars = ["PINECONE_API_KEY"]
    if embedding_type == "openai":
        required_env_vars.append("OPENAI_API_KEY")
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        if embedding_type == "local":
            print("💡 Tip: Using local embeddings only requires PINECONE_API_KEY")
        return
    
    try:
        # Initialize searcher with same configuration as processing
        searcher = SegmentSearcher(
            embedding_type=embedding_type,
            local_model_name=local_model
        )
        
        # Perform search
        results = searcher.search_segments(query, top_k=5)
        
        # Enrich with full text if requested
        if show_full_text:
            results = searcher.enrich_results_with_full_text(results)
        
        # Display results
        if show_full_text:
            print(f"✅ Results enriched with full text from transcript files")
        searcher.display_results(results, show_full_text=show_full_text)
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        print(f"🔍 Debug info: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 