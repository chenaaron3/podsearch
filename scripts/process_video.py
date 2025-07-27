#!/usr/bin/env python3
"""
Video Processing Pipeline

Extracts transcripts using Whisper, groups segments based on semantic similarity,
analyzes emotions using GoEmotions, and stores embeddings in Pinecone for semantic search.

Features:
- Whisper-based transcript extraction with timestamps
- Semantic segmentation (1-2 minute segments)
- GoEmotions-based emotion detection for each segment
- OpenAI embeddings for similarity analysis
- Pinecone vector storage
- Database integration for video metadata
- Batch processing of downloaded videos

Usage:
    python process_video.py --video-id <video_id>
    python process_video.py --playlist-id <playlist_id>
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import whisper
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import ffmpeg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Import database manager
from database import (
    DatabaseManager, 
    TranscriptData, 
    TranscriptSegment, 
    SemanticSegment,
    SegmentWithEmotion,
    SegmentWithEmbedding,
    PineconeMetadata,
    PineconeVector,
    Video,
    VideoStatus
)

# Load environment variables
load_dotenv()

class VideoProcessor:
    def __init__(self, 
                 downloads_dir: str = "./downloads",
                 output_dir: str = "./processed",
                 pinecone_index_name: str = "video-segments",
                 embedding_type: str = "openai",
                 local_model_name: str = "all-MiniLM-L6-v2",
                 enable_emotion_analysis: bool = True):
        """
        Initialize the video processor.
        
        Args:
            downloads_dir: Directory containing downloaded videos
            output_dir: Directory to save processed results
            pinecone_index_name: Name for the Pinecone index
            embedding_type: "local" or "openai" for embedding method
            local_model_name: Name of local sentence transformer model
            enable_emotion_analysis: Whether to enable emotion detection using GoEmotions
        """
        self.downloads_dir = Path(downloads_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized storage
        (self.output_dir / "transcripts").mkdir(exist_ok=True)
        (self.output_dir / "segments").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "emotions").mkdir(exist_ok=True)
        
        # Initialize database manager
        self.db_manager = DatabaseManager()
        
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
            self.embedding_dimensions = 3072  # OpenAI text-embedding-3-large dimensions
            print("✅ OpenAI client initialized")
        else:
            raise ValueError(f"Invalid embedding_type: {embedding_type}. Use 'local' or 'openai'")
        
        # Emotion analysis configuration
        self.enable_emotion_analysis = enable_emotion_analysis
        if enable_emotion_analysis:
            print("🔄 Loading GoEmotions emotion analysis model...")
            try:
                self.emotion_classifier = pipeline(
                    task="text-classification",
                    model="SamLowe/roberta-base-go_emotions",
                    top_k=None,
                    device=-1  # Use CPU; set to 0 for GPU if available
                )
                print("✅ GoEmotions model loaded successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not load emotion model: {e}")
                print("   Emotion analysis will be disabled")
                self.enable_emotion_analysis = False
                self.emotion_classifier = None
        else:
            self.emotion_classifier = None
            print("ℹ️ Emotion analysis disabled")
        
        print("🔄 Connecting to Pinecone...")
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = pinecone_index_name
        self.index = None
        
        # Segment parameters
        self.target_segment_duration = 45   # 45 seconds target
        self.min_segment_duration = 30     # 30 seconds minimum  
        self.max_segment_duration = 60     # 60 seconds maximum
        self.similarity_threshold = 0.85   # Threshold for grouping similar segments
        
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

    def extract_transcript(self, video: Video, force_reprocess: bool = False) -> Optional[TranscriptData]:
        """
        Extract transcript using Whisper with timestamps, using database video metadata.
        
        Args:
            video_id: Database ID of the video
            force_reprocess: If True, reprocess even if transcript exists
            
        Returns:
            Dictionary containing transcript data with timestamps or None if failed
        """
        # Get video from database using DAO
        if not video.local_file_path:
            print(f"❌ No local file path for video {video.id}")
            return None
            
        video_path = Path(video.local_file_path)
        if not video_path.exists():
            print(f"❌ Video file not found: {video_path}")
            return None
        
        transcript_file = self.output_dir / "transcripts" / f"{video.id}_transcript.json"
        
        # Check if transcript already exists
        if transcript_file.exists() and not force_reprocess:
            print(f"📄 Loading existing transcript: {video.title}")
            try:
                with open(transcript_file, "r", encoding="utf-8") as f:
                    transcript_data = json.load(f)
                
                # Validate the transcript has required fields
                if all(key in transcript_data for key in ["segments"]):
                    print(f"✅ Transcript loaded: {len(transcript_data['segments'])} segments")
                    if not video.transcript_id:
                        self.db_manager.save_transcript_data(video.id, transcript_data)
                    return transcript_data
                else:
                    print("⚠️ Existing transcript incomplete, reprocessing...")
                    
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️ Error loading existing transcript: {e}, reprocessing...")
        
        # Process with Whisper if no valid transcript exists
        print(f"🎤 Extracting transcript with Whisper: {video.title}")
        
        try:
            # Extract audio and transcribe
            result = self.whisper_model.transcribe(
                str(video_path),
                language="en",
                word_timestamps=True,
                verbose=False
            )
            
            # Structure the transcript data using database metadata
            transcript_data = {
                "language": result.get("language", "en"),
                "segments": [],
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
            
            # Save transcript to file
            with open(transcript_file, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
            # Save transcript to database using DAO
            self.db_manager.save_transcript_data(video.id, transcript_data)
            
            print(f"✅ Transcript extracted and saved: {len(transcript_data['segments'])} segments")
            return transcript_data
            
        except Exception as e:
            print(f"❌ Error extracting transcript: {e}")
            return None

    def create_semantic_segments(self, video: Video, transcript_data: TranscriptData, force_reprocess: bool = False) -> List[SemanticSegment]:
        """
        Create semantic segments based on topic changes and target duration.
        
        Args:
            transcript_data: Output from extract_transcript
            force_reprocess: If True, reprocess even if segments exist
            
        Returns:
            List of semantic segments
        """
        segments_file = self.output_dir / "segments" / f"{video.id}_segments.json"
        
        # Check if segments already exist
        if segments_file.exists() and not force_reprocess:
            print(f"📄 Loading existing segments: {video.title}")
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
        }
        
        for i, segment in enumerate(segments[1:], 1):
            # Skip empty segments
            if not segment.get("text", "").strip():
                continue
                
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
                      segment["text"].strip() and segment["text"].strip()[0].isupper()):
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
                }
            else:
                # Extend current segment
                current_segment["end_time"] = segment["end"]
                current_segment["text"] += " " + segment["text"]
        
        # Add the last segment
        if current_segment["text"]:
            semantic_segments.append(current_segment)
        
        # Post-process segments
        processed_segments = []
        for i, seg in enumerate(semantic_segments):
            processed_segment = {
                "segment_id": i + 1,
                "video_id": video.id,
                "video_name": video.title,
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "duration": seg["end_time"] - seg["start_time"],
                "text": seg["text"].strip(),
                "timestamp_readable": f"{int(seg['start_time']//60):02d}:{int(seg['start_time']%60):02d} - {int(seg['end_time']//60):02d}:{int(seg['end_time']%60):02d}"
            }
            processed_segments.append(processed_segment)
        
        # Save segments
        with open(segments_file, "w", encoding="utf-8") as f:
            json.dump(processed_segments, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created and saved {len(processed_segments)} semantic segments")
        return processed_segments

    def analyze_emotions(self, video: Video, segments: List[SemanticSegment], force_reprocess: bool = False) -> List[SegmentWithEmotion]:
        """
        Analyze emotions for semantic segments using GoEmotions.
        
        Args:
            segments: List of semantic segments
            force_reprocess: If True, reprocess even if emotions exist
            
        Returns:
            Segments with emotion analysis added
        """
        if not segments or not self.enable_emotion_analysis:
            print("ℹ️ Emotion analysis skipped (disabled or no segments)")
            return segments
            
        emotions_file = self.output_dir / "emotions" / f"{video.id}_emotions.json"
        
        # Check if emotions already exist
        if emotions_file.exists() and not force_reprocess:
            print(f"📄 Loading existing emotion analysis: {video.title}")
            try:
                with open(emotions_file, "r", encoding="utf-8") as f:
                    existing_emotions = json.load(f)
                
                # Validate emotions have required fields and match current segments
                if (existing_emotions and 
                    len(existing_emotions) == len(segments) and
                    all(key in existing_emotions[0] for key in ["primary_emotion", "primary_emotion_score"])):
                    print(f"✅ Emotion analysis loaded: {len(existing_emotions)} segments")
                    return existing_emotions
                else:
                    print("⚠️ Existing emotion analysis incompatible or incomplete, reprocessing...")
                    
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️ Error loading existing emotion analysis: {e}, reprocessing...")
        
        print(f"😊 Analyzing emotions for {len(segments)} segments using GoEmotions...")
        
        try:
            # Prepare texts for emotion analysis
            texts = [seg["text"] for seg in segments]
            
            # Process in batches to manage memory and API limits
            batch_size = 10
            all_emotion_results = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                print(f"  Processing emotion batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                # Get emotion predictions for batch
                batch_results = self.emotion_classifier(batch_texts)
                all_emotion_results.extend(batch_results)
            
            # Add emotion analysis to segments
            segments_with_emotions = []
            for i, segment in enumerate(segments):
                segment_with_emotions = segment.copy()
                emotion_data = all_emotion_results[i]
                
                # Process emotion results
                emotions = {}
                emotion_scores = {}
                for emotion_result in emotion_data:
                    emotion_label = emotion_result['label']
                    emotion_score = emotion_result['score']
                    emotions[emotion_label] = emotion_score
                    emotion_scores[emotion_label] = float(emotion_score)
                
                # Find primary emotion (highest score)
                primary_emotion = max(emotions.items(), key=lambda x: x[1])
                
                # Add emotion data to segment
                segment_with_emotions["primary_emotion"] = primary_emotion[0]
                segment_with_emotions["primary_emotion_score"] = float(primary_emotion[1])

                segments_with_emotions.append(segment_with_emotions)
            
            # Save emotion analysis
            with open(emotions_file, "w", encoding="utf-8") as f:
                json.dump(segments_with_emotions, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Emotion analysis completed and saved for {len(segments_with_emotions)} segments")
            
            # Print emotion summary
            primary_emotions = [seg["primary_emotion"] for seg in segments_with_emotions]
            emotion_counts = {}
            for emotion in primary_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            print(f"📊 Primary emotion distribution:")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(segments_with_emotions)) * 100
                print(f"   {emotion}: {count} segments ({percentage:.1f}%)")
            
            return segments_with_emotions
            
        except Exception as e:
            print(f"❌ Error analyzing emotions: {e}")
            print("⚠️ Continuing without emotion analysis...")
            return segments

    def generate_embeddings(self, video: Video, segments: List[SegmentWithEmotion], force_reprocess: bool = False) -> List[SegmentWithEmbedding]:
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
            
        embeddings_file = self.output_dir / "embeddings" / f"{video.id}_embeddings.json"
        
        # Check if embeddings already exist
        if embeddings_file.exists() and not force_reprocess:
            print(f"📄 Loading existing embeddings: {video.title}")
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
                        model="text-embedding-3-large",
                        input=batch_texts
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                
                model_name = "text-embedding-3-large"
            
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

    def store_in_pinecone(self, video: Video, segments_with_embeddings: List[SegmentWithEmbedding]):
        """
        Store segments and embeddings in Pinecone.
        
        Args:
            segments_with_embeddings: Segments with their embeddings and emotion analysis
        """
        print(f"📌 Storing {len(segments_with_embeddings)} segments in Pinecone...")
        
        try:
            # Mark video as embedding
            if video.status in [VideoStatus.EMBEDDED, VideoStatus.FINISHED]:
                print(f"🔄 Video already embedded, skipping...")
                return

            # Prepare vectors for upsert
            vectors: List[PineconeVector] = []
            for segment in segments_with_embeddings:
                # Create vector ID using video_id for consistency
                video_id = segment['video_id']
                vector_id = f"video_{video_id}_segment_{segment['segment_id']}"
                
                metadata: PineconeMetadata = {
                    "video_id": segment["video_id"],
                    "video_name": segment["video_name"],
                    "segment_id": segment["segment_id"],
                    "start_time": float(segment["start_time"]),
                    "end_time": float(segment["end_time"]),
                    "duration": float(segment["duration"]),
                    "timestamp_readable": segment["timestamp_readable"],
                    "primary_emotion": segment["primary_emotion"],
                    "primary_emotion_score": float(segment["primary_emotion_score"]),
                }
                    
                vector: PineconeVector = {
                    "id": vector_id,
                    "values": [float(x) for x in segment["embedding"]],  # Convert to native Python floats
                    "metadata": metadata
                }
                vectors.append(vector)
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                print(f"  Uploaded batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            print(f"✅ Stored {len(vectors)} vectors in Pinecone")
            
            # Print metadata summary if emotions are included
            emotions_in_pinecone = [seg["primary_emotion"] for seg in segments_with_embeddings if "primary_emotion" in seg]
            unique_emotions = set(emotions_in_pinecone)
            print(f"📊 Stored segments with {len(unique_emotions)} different primary emotions: {sorted(unique_emotions)}")
            
            # Mark video as done after embeddings are stored
            self.db_manager.update_video_status(video.id, VideoStatus.EMBEDDED) 
        except Exception as e:
            print(f"❌ Error storing in Pinecone: {e}")
            raise

    def process_single_video(self, video_id: int, force_reprocess: bool = False) -> bool:
        """
        Process a single video through the complete pipeline.
        
        Args:
            video_id: Database ID of the video to process
            force_reprocess: If True, reprocess all steps even if cached results exist
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n🎬 Processing video: {video_id}")
        if force_reprocess:
            print("🔄 Force reprocessing enabled - ignoring cached results")
        
        try:
            video = self.db_manager.get_video_by_id(video_id)
            # Step 1: Extract transcript
            transcript_data = self.extract_transcript(video, force_reprocess)
            
            if not transcript_data:
                print(f"❌ Failed to extract transcript for video {video_id}")
                return False
            
            # Step 2: Create semantic segments
            segments = self.create_semantic_segments(video, transcript_data, force_reprocess)
            
            if not segments:
                print("⚠️ No segments created, skipping video")
                return False
            
            # Step 3: Analyze emotions (new step!)
            segments_with_emotions = self.analyze_emotions(video, segments, force_reprocess)
            
            # Step 4: Generate embeddings
            segments_with_embeddings = self.generate_embeddings(video, segments_with_emotions, force_reprocess)
            
            # Step 6: Store in Pinecone
            self.store_in_pinecone(video, segments_with_embeddings)
            
            # Save similarity analysis
            analysis_file = self.output_dir / f"{video.title}_analysis.json"
            analysis_data = {
                "video_id": video_id,
                "total_segments": len(segments),
                "average_segment_duration": np.mean([seg["duration"] for seg in segments]),
                "emotion_analysis_enabled": self.enable_emotion_analysis,
                "processed_at": datetime.now().isoformat()
            }
            
            # Add emotion analysis summary if available
            if segments_with_emotions and "primary_emotion" in segments_with_emotions[0]:
                primary_emotions = [seg["primary_emotion"] for seg in segments_with_emotions]
                emotion_counts = {}
                for emotion in primary_emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                analysis_data["emotion_distribution"] = emotion_counts
                analysis_data["most_common_emotion"] = max(emotion_counts.items(), key=lambda x: x[1])[0]
            
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, indent=2)
            
            self.db_manager.update_video_status(video.id, VideoStatus.FINISHED)
            print(f"✅ Successfully processed video {video_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing video {video_id}: {e}")
            return False