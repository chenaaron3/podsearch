#!/usr/bin/env python3
"""
Video Processing Pipeline

Extracts transcripts using Whisper, groups segments based on semantic similarity,
and stores embeddings in Pinecone for semantic search.

Features:
- Whisper-based transcript extraction with timestamps
- Semantic segmentation (1-2 minute segments)
- OpenAI embeddings for similarity analysis
- Pinecone vector storage
- Batch processing of downloaded videos

Usage:
    python process_video.py
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

import whisper
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import ffmpeg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class VideoProcessor:
    def __init__(self, 
                 downloads_dir: str = "./downloads",
                 output_dir: str = "./processed",
                 pinecone_index_name: str = "video-segments",
                 embedding_type: str = "local",
                 local_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the video processor.
        
        Args:
            downloads_dir: Directory containing downloaded videos
            output_dir: Directory to save processed results
            pinecone_index_name: Name for the Pinecone index
            embedding_type: "local" or "openai" for embedding method
            local_model_name: Name of local sentence transformer model
        """
        self.downloads_dir = Path(downloads_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized storage
        (self.output_dir / "transcripts").mkdir(exist_ok=True)
        (self.output_dir / "segments").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        
        # Initialize models and clients
        print("🔄 Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        
        # Embedding configuration
        self.embedding_type = embedding_type
        self.local_model_name = local_model_name
        
        # Initialize embedding model based on type
        if embedding_type == "local":
            print(f"🔄 Loading local embedding model: {local_model_name}...")
            self.embedding_model = SentenceTransformer(local_model_name)
            self.embedding_dimensions = self.embedding_model.get_sentence_embedding_dimension()
            self.openai_client = None
            print(f"✅ Local model loaded with {self.embedding_dimensions} dimensions")
        elif embedding_type == "openai":
            print("🔄 Initializing OpenAI client...")
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.embedding_model = None
            self.embedding_dimensions = 1536  # OpenAI text-embedding-3-small dimensions
            print("✅ OpenAI client initialized")
        else:
            raise ValueError(f"Invalid embedding_type: {embedding_type}. Use 'local' or 'openai'")
        
        print("🔄 Connecting to Pinecone...")
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = pinecone_index_name
        self.index = None
        
        # Segment parameters
        self.target_segment_duration = 90  # 1.5 minutes in seconds
        self.min_segment_duration = 60    # 1 minute minimum
        self.max_segment_duration = 120   # 2 minutes maximum
        self.similarity_threshold = 0.85  # Threshold for grouping similar segments
        
        self._setup_pinecone_index()

    def _setup_pinecone_index(self):
        """Set up Pinecone index for storing embeddings."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"🔄 Creating new Pinecone index: {self.index_name}")
                print(f"   Dimensions: {self.embedding_dimensions}")
                print(f"   Embedding type: {self.embedding_type}")
                
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimensions,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"✅ Created Pinecone index: {self.index_name}")
            else:
                print(f"✅ Using existing Pinecone index: {self.index_name}")
                # Verify dimensions match
                index_stats = self.pinecone_client.describe_index(self.index_name)
                if hasattr(index_stats, 'dimension') and index_stats.dimension != self.embedding_dimensions:
                    print(f"⚠️  Warning: Index dimension ({index_stats.dimension}) doesn't match model ({self.embedding_dimensions})")
            
            self.index = self.pinecone_client.Index(self.index_name)
            
        except Exception as e:
            print(f"❌ Error setting up Pinecone index: {e}")
            raise

    def extract_video_url_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract video URL from filename if it contains YouTube video ID pattern.
        This is a fallback - ideally you'd store the URL during download.
        """
        # For now, return a placeholder. In production, you'd want to store
        # the original URL during the download process.
        video_id = re.search(r'[a-zA-Z0-9_-]{11}', filename)
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id.group()}"
        return f"unknown_video_{hashlib.md5(filename.encode()).hexdigest()[:8]}"

    @staticmethod
    def sanitize_vector_id(text: str) -> str:
        """Sanitize text to be ASCII-only for Pinecone vector IDs."""
        # Convert to ASCII, replacing non-ASCII characters
        ascii_text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Replace any remaining problematic characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', ascii_text)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it's not empty and not too long
        if not sanitized:
            sanitized = "video"
        
        return sanitized[:100]  # Limit length for Pinecone

    def extract_transcript(self, video_path: Path, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Extract transcript using Whisper with timestamps, or load existing if available.
        
        Args:
            video_path: Path to the video file
            force_reprocess: If True, reprocess even if transcript exists
            
        Returns:
            Dictionary containing transcript data with timestamps
        """
        transcript_file = self.output_dir / "transcripts" / f"{video_path.stem}_transcript.json"
        
        # Check if transcript already exists
        if transcript_file.exists() and not force_reprocess:
            print(f"📄 Loading existing transcript: {video_path.name}")
            try:
                with open(transcript_file, "r", encoding="utf-8") as f:
                    transcript_data = json.load(f)
                
                # Validate the transcript has required fields
                if all(key in transcript_data for key in ["video_path", "segments", "full_text"]):
                    print(f"✅ Transcript loaded: {len(transcript_data['segments'])} segments")
                    return transcript_data
                else:
                    print("⚠️ Existing transcript incomplete, reprocessing...")
                    
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️ Error loading existing transcript: {e}, reprocessing...")
        
        # Process with Whisper if no valid transcript exists
        print(f"🎤 Extracting transcript with Whisper: {video_path.name}")
        
        try:
            # Extract audio and transcribe
            result = self.whisper_model.transcribe(
                str(video_path),
                language="en",
                word_timestamps=True,
                verbose=False
            )
            
            # Structure the transcript data
            transcript_data = {
                "video_path": str(video_path),
                "video_name": video_path.name,
                "video_url": self.extract_video_url_from_filename(video_path.name),
                "duration": result.get("duration", 0),
                "language": result.get("language", "en"),
                "segments": [],
                "full_text": result["text"],
                "processed_at": datetime.now().isoformat()
            }
            
            # Process segments with word-level timestamps
            for segment in result["segments"]:
                segment_data = {
                    "id": segment["id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "words": segment.get("words", [])
                }
                transcript_data["segments"].append(segment_data)
            
            # Save transcript
            with open(transcript_file, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Transcript extracted and saved: {len(transcript_data['segments'])} segments")
            return transcript_data
            
        except Exception as e:
            print(f"❌ Error extracting transcript: {e}")
            raise

    def create_semantic_segments(self, transcript_data: Dict[str, Any], force_reprocess: bool = False) -> List[Dict[str, Any]]:
        """
        Create semantic segments based on topic changes and target duration.
        
        Args:
            transcript_data: Output from extract_transcript
            force_reprocess: If True, reprocess even if segments exist
            
        Returns:
            List of semantic segments
        """
        video_path = Path(transcript_data["video_path"])
        segments_file = self.output_dir / "segments" / f"{video_path.stem}_segments.json"
        
        # Check if segments already exist
        if segments_file.exists() and not force_reprocess:
            print(f"📄 Loading existing segments: {video_path.name}")
            try:
                with open(segments_file, "r", encoding="utf-8") as f:
                    existing_segments = json.load(f)
                
                # Validate segments have required fields
                if existing_segments and all(key in existing_segments[0] for key in ["segment_id", "start_time", "end_time", "text"]):
                    print(f"✅ Segments loaded: {len(existing_segments)} segments")
                    return existing_segments
                else:
                    print("⚠️ Existing segments incomplete, reprocessing...")
                    
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️ Error loading existing segments: {e}, reprocessing...")
        
        print(f"🔀 Creating semantic segments...")
        
        segments = transcript_data["segments"]
        if not segments:
            return []
        
        semantic_segments = []
        current_segment = {
            "start_time": segments[0]["start"],
            "end_time": segments[0]["end"],
            "text": segments[0]["text"],
            "source_segments": [segments[0]["id"]]
        }
        
        for i, segment in enumerate(segments[1:], 1):
            current_duration = current_segment["end_time"] - current_segment["start_time"]
            
            # Check if we should start a new semantic segment
            should_split = False
            
            # Duration-based splitting
            if current_duration >= self.max_segment_duration:
                should_split = True
            elif current_duration >= self.min_segment_duration:
                # Check for semantic breaks (topic changes)
                # Simple heuristic: look for longer pauses or sentence endings
                time_gap = segment["start"] - current_segment["end_time"]
                if time_gap > 2.0:  # 2+ second pause suggests topic change
                    should_split = True
                elif (current_segment["text"].endswith(('.', '!', '?')) and 
                      segment["text"].strip()[0].isupper()):
                    # Sentence boundary with capital letter (new topic)
                    should_split = True
            
            if should_split and current_duration >= self.min_segment_duration:
                # Finalize current segment
                semantic_segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": segment["text"],
                    "source_segments": [segment["id"]]
                }
            else:
                # Extend current segment
                current_segment["end_time"] = segment["end"]
                current_segment["text"] += " " + segment["text"]
                current_segment["source_segments"].append(segment["id"])
        
        # Add the last segment
        if current_segment["text"]:
            semantic_segments.append(current_segment)
        
        # Post-process segments
        processed_segments = []
        for i, seg in enumerate(semantic_segments):
            processed_segment = {
                "segment_id": i + 1,
                "video_path": transcript_data["video_path"],
                "video_name": transcript_data["video_name"],
                "video_url": transcript_data["video_url"],
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "duration": seg["end_time"] - seg["start_time"],
                "text": seg["text"].strip(),
                "source_segments": seg["source_segments"],
                "timestamp_readable": f"{int(seg['start_time']//60):02d}:{int(seg['start_time']%60):02d} - {int(seg['end_time']//60):02d}:{int(seg['end_time']%60):02d}"
            }
            processed_segments.append(processed_segment)
        
        # Save segments
        with open(segments_file, "w", encoding="utf-8") as f:
            json.dump(processed_segments, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created and saved {len(processed_segments)} semantic segments")
        return processed_segments

    def generate_embeddings(self, segments: List[Dict[str, Any]], force_reprocess: bool = False) -> List[Dict[str, Any]]:
        """
        Generate embeddings for semantic segments (local or OpenAI).
        
        Args:
            segments: List of semantic segments
            force_reprocess: If True, reprocess even if embeddings exist
            
        Returns:
            Segments with embeddings added
        """
        if not segments:
            return []
            
        video_path = Path(segments[0]['video_path'])
        embeddings_file = self.output_dir / "embeddings" / f"{video_path.stem}_embeddings.json"
        
        # Check if embeddings already exist
        if embeddings_file.exists() and not force_reprocess:
            print(f"📄 Loading existing embeddings: {video_path.name}")
            try:
                with open(embeddings_file, "r", encoding="utf-8") as f:
                    existing_embeddings = json.load(f)
                
                # Validate embeddings have required fields and match current embedding type
                if (existing_embeddings and 
                    len(existing_embeddings) == len(segments) and
                    all(key in existing_embeddings[0] for key in ["embedding", "embedding_type", "text"]) and
                    existing_embeddings[0].get("embedding_type") == self.embedding_type):
                    print(f"✅ Embeddings loaded: {len(existing_embeddings)} segments ({self.embedding_type})")
                    return existing_embeddings
                else:
                    print("⚠️ Existing embeddings incompatible or incomplete, reprocessing...")
                    
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️ Error loading existing embeddings: {e}, reprocessing...")
        
        print(f"🧮 Generating {self.embedding_type} embeddings for {len(segments)} segments...")
        
        # Prepare texts for embedding
        texts = [seg["text"] for seg in segments]
        
        try:
            if self.embedding_type == "local":
                # Use local sentence transformer model
                print(f"   Using model: {self.local_model_name}")
                
                # Generate embeddings (sentence-transformers handles batching automatically)
                all_embeddings = self.embedding_model.encode(
                    texts,
                    batch_size=32,  # Adjust based on GPU memory
                    show_progress_bar=True,
                    convert_to_numpy=True
                ).tolist()
                
                model_name = self.local_model_name
                
            elif self.embedding_type == "openai":
                # Use OpenAI API with batching for rate limits
                batch_size = 100
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    print(f"  Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                    
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=batch_texts
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                
                model_name = "text-embedding-3-small"
            
            # Add embeddings to segments
            segments_with_embeddings = []
            for i, segment in enumerate(segments):
                segment_with_embedding = segment.copy()
                # Ensure embedding is a list of native Python floats
                embedding = all_embeddings[i]
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                elif not isinstance(embedding, list):
                    embedding = list(embedding)
                segment_with_embedding["embedding"] = embedding
                segment_with_embedding["embedding_model"] = model_name
                segment_with_embedding["embedding_type"] = self.embedding_type
                segment_with_embedding["embedding_dimensions"] = len(embedding)
                segments_with_embeddings.append(segment_with_embedding)
            
            # Save embeddings
            with open(embeddings_file, "w", encoding="utf-8") as f:
                json.dump(segments_with_embeddings, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Generated and saved {self.embedding_type} embeddings with {len(all_embeddings[0])} dimensions")
            return segments_with_embeddings
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            raise

    def store_in_pinecone(self, segments_with_embeddings: List[Dict[str, Any]]):
        """
        Store segments and embeddings in Pinecone.
        
        Args:
            segments_with_embeddings: Segments with their embeddings
        """
        print(f"📌 Storing {len(segments_with_embeddings)} segments in Pinecone...")
        
        try:
            # Prepare vectors for upsert
            vectors = []
            for segment in segments_with_embeddings:
                # Sanitize vector ID to be ASCII-only for Pinecone
                video_stem = Path(segment['video_path']).stem
                sanitized_stem = self.sanitize_vector_id(video_stem)
                vector_id = f"{sanitized_stem}_{segment['segment_id']}"
                
                metadata = {
                    "video_name": str(segment["video_name"]),
                    "video_url": str(segment["video_url"]),
                    "segment_id": int(segment["segment_id"]),
                    "start_time": float(segment["start_time"]),
                    "end_time": float(segment["end_time"]),
                    "duration": float(segment["duration"]),
                    "timestamp_readable": str(segment["timestamp_readable"]),
                    "full_text_length": int(len(segment["text"]))
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": [float(x) for x in segment["embedding"]],  # Convert to native Python floats
                    "metadata": metadata
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                print(f"  Uploaded batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            print(f"✅ Stored {len(vectors)} vectors in Pinecone")
            
        except Exception as e:
            print(f"❌ Error storing in Pinecone: {e}")
            raise

    def find_similar_segments(self, segments_with_embeddings: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Find groups of similar segments based on embedding similarity.
        
        Args:
            segments_with_embeddings: Segments with their embeddings
            
        Returns:
            List of lists, where each inner list contains indices of similar segments
        """
        print(f"🔍 Finding similar segments...")
        
        if len(segments_with_embeddings) < 2:
            return [[0]] if segments_with_embeddings else []
        
        # Extract embeddings
        embeddings = np.array([seg["embedding"] for seg in segments_with_embeddings])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find similar segments
        similar_groups = []
        processed = set()
        
        for i in range(len(segments_with_embeddings)):
            if i in processed:
                continue
            
            # Find segments similar to current one
            similar_indices = []
            for j in range(len(segments_with_embeddings)):
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    similar_indices.append(j)
                    processed.add(j)
            
            if similar_indices:
                similar_groups.append(similar_indices)
        
        print(f"✅ Found {len(similar_groups)} groups of similar segments")
        return similar_groups

    def process_single_video(self, video_path: Path, force_reprocess: bool = False) -> bool:
        """
        Process a single video through the complete pipeline.
        
        Args:
            video_path: Path to the video file
            force_reprocess: If True, reprocess all steps even if cached results exist
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n🎬 Processing video: {video_path.name}")
        if force_reprocess:
            print("🔄 Force reprocessing enabled - ignoring cached results")
        
        try:
            # Step 1: Extract transcript
            transcript_data = self.extract_transcript(video_path, force_reprocess)
            
            # Step 2: Create semantic segments
            segments = self.create_semantic_segments(transcript_data, force_reprocess)
            
            if not segments:
                print("⚠️ No segments created, skipping video")
                return False
            
            # Step 3: Generate embeddings
            segments_with_embeddings = self.generate_embeddings(segments, force_reprocess)
            
            # Step 4: Find similar segments (for analysis)
            similar_groups = self.find_similar_segments(segments_with_embeddings)
            
            # Step 5: Store in Pinecone
            self.store_in_pinecone(segments_with_embeddings)
            
            # Save similarity analysis
            analysis_file = self.output_dir / f"{video_path.stem}_analysis.json"
            analysis_data = {
                "video_name": video_path.name,
                "total_segments": len(segments),
                "similar_groups": similar_groups,
                "average_segment_duration": np.mean([seg["duration"] for seg in segments]),
                "processed_at": datetime.now().isoformat()
            }
            
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, indent=2)
            
            print(f"✅ Successfully processed {video_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {video_path.name}: {e}")
            return False

    def process_all_videos(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process all videos in the downloads directory.
        
        Args:
            force_reprocess: If True, reprocess all steps even if cached results exist
        
        Returns:
            Summary of processing results
        """
        print(f"🚀 Starting batch processing of videos in {self.downloads_dir}")
        if force_reprocess:
            print("🔄 Force reprocessing enabled - ignoring all cached results")
        
        # Find all video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_files = [
            f for f in self.downloads_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in video_extensions and not f.name.endswith('.part')
        ]
        
        print(f"📁 Found {len(video_files)} video files to process")
        
        if not video_files:
            print("⚠️ No video files found")
            return {"total_videos": 0, "successful": 0, "failed": 0}
        
        # Process each video
        successful = 0
        failed = 0
        results = []
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n📹 Processing video {i}/{len(video_files)}")
            
            if self.process_single_video(video_file, force_reprocess):
                successful += 1
                results.append({"video": video_file.name, "status": "success"})
            else:
                failed += 1
                results.append({"video": video_file.name, "status": "failed"})
        
        # Save batch processing summary
        summary = {
            "total_videos": len(video_files),
            "successful": successful,
            "failed": failed,
            "results": results,
            "processed_at": datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / "batch_processing_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 Batch processing complete!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success rate: {successful/len(video_files)*100:.1f}%")
        
        return summary

def main():
    """Main function to run the video processing pipeline."""
    print("🎬 Video Processing Pipeline")
    print("=" * 50)
    
    # Configuration options
    embedding_type = os.getenv("EMBEDDING_TYPE", "local").lower()
    local_model = os.getenv("LOCAL_MODEL", "all-MiniLM-L6-v2")
    force_reprocess = os.getenv("FORCE_REPROCESS", "false").lower() in ("true", "1", "yes")
    
    print(f"🔧 Configuration:")
    print(f"   Embedding type: {embedding_type}")
    if embedding_type == "local":
        print(f"   Local model: {local_model}")
    print(f"   Force reprocess: {force_reprocess}")
    print()
    
    # Check required environment variables based on embedding type
    required_env_vars = ["PINECONE_API_KEY"]
    if embedding_type == "openai":
        required_env_vars.append("OPENAI_API_KEY")
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment")
        if embedding_type == "local":
            print("💡 Tip: Using local embeddings only requires PINECONE_API_KEY")
        return
    
    try:
        # Initialize processor with configuration
        processor = VideoProcessor(
            embedding_type=embedding_type,
            local_model_name=local_model
        )
        
        # Process all videos
        summary = processor.process_all_videos(force_reprocess)
        
        print(f"\n📋 Processing Summary:")
        print(f"   Total videos: {summary['total_videos']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Embedding type: {embedding_type}")
        
        if summary['successful'] > 0:
            print(f"\n🔍 You can now search your video segments using Pinecone!")
            print(f"   Index name: {processor.index_name}")
            print(f"   Embedding dimensions: {processor.embedding_dimensions}")
            print(f"   Search with: python search_segments.py \"your query\"")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        print(f"🔍 Debug info: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
