#!/usr/bin/env python3
"""
Database models and connection for the video processing pipeline.
Matches the Drizzle schema in schema.ts
"""

import os
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, TypedDict
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Index, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import OperationalError, DisconnectionError
from dotenv import load_dotenv

load_dotenv()

# Type definitions for transcript data
class TranscriptSegment(TypedDict):
    id: int
    start: float
    end: float
    text: str
    words: Optional[List[Dict[str, Any]]]

class TranscriptData(TypedDict):
    language: str
    segments: List[TranscriptSegment]
    processed_at: str

class SemanticSegment(TypedDict):
    segment_id: int
    video_id: int
    video_name: str
    start_time: float
    end_time: float
    duration: float
    text: str
    timestamp_readable: str
    source_segments: List[str]

class EmotionScore(TypedDict):
    emotion: str
    score: float

class SegmentWithEmotion(SemanticSegment):
    emotions: Dict[str, float]
    primary_emotion: str
    primary_emotion_score: float

class SegmentWithEmbedding(SegmentWithEmotion):
    embedding: List[float]
    embedding_model: str
    embedding_type: str
    embedding_dimensions: int

class PineconeMetadata(TypedDict):
    video_id: int
    video_name: str
    segment_id: int
    start_time: float
    end_time: float
    duration: float
    timestamp_readable: str
    primary_emotion: Optional[str]
    primary_emotion_score: Optional[float]
    source_segments: List[str]

class PineconeVector(TypedDict):
    id: str
    values: List[float]
    metadata: PineconeMetadata

Base = declarative_base()

class VideoStatus:
    """Video processing status constants matching the Drizzle schema."""
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    EMBEDDED = "embedded"
    FINISHED = "finished"
    FAILED = "failed"
    
    @classmethod
    def all_values(cls):
        """Get all status values as a list."""
        return [cls.PENDING, cls.DOWNLOADED, cls.EMBEDDED, cls.FINISHED, cls.FAILED]

class Playlist(Base):
    __tablename__ = "podsearch_playlist" 
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    youtube_id = Column("youtubeId", String(255), nullable=False, unique=True)  
    title = Column(String(500), nullable=False)
    description = Column(Text)
    channel_name = Column("channelName", String(255))
    channel_id = Column("channelId", String(255))
    url = Column(String(500), nullable=False)
    total_videos = Column("totalVideos", Integer, default=0)
    created_at = Column("createdAt", DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column("updatedAt", DateTime(timezone=True), onupdate=func.now())
    last_sync_at = Column("lastSyncAt", DateTime(timezone=True))
    
    # Relationships
    videos = relationship("Video", back_populates="playlist", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("playlist_youtube_id_idx", "youtubeId"),
        Index("playlist_channel_id_idx", "channelId"),
    )

class Transcript(Base):
    __tablename__ = "podsearch_transcript"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    language = Column(String(10), default="en")
    segments = Column(JSONB)  # Array of transcript segments with timestamps
    processing_metadata = Column("processingMetadata", JSONB)  # Metadata about processing
    created_at = Column("createdAt", DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column("updatedAt", DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    video = relationship("Video", back_populates="transcript", uselist=False)
    
    # Indexes
    __table_args__ = (
        Index("transcript_language_idx", "language"),
    )

class Video(Base):
    __tablename__ = "podsearch_video"
    
    id = Column(Integer, primary_key=True, autoincrement=True)  
    youtube_id = Column("youtubeId", String(255), nullable=False, unique=True)
    playlist_id = Column("playlistId", Integer, ForeignKey("podsearch_playlist.id", ondelete="CASCADE"))
    transcript_id = Column("transcriptId", Integer, ForeignKey("podsearch_transcript.id", ondelete="SET NULL"))
    title = Column(String(500), nullable=False)
    description = Column(Text)
    duration = Column(Integer)  # duration in seconds
    published_at = Column("publishedAt", DateTime(timezone=True))
    thumbnail_url = Column("thumbnailUrl", String(500))
    url = Column(String(500), nullable=False)
    local_file_path = Column("localFilePath", String(1000))
    status = Column("status", String(20), default=VideoStatus.PENDING, nullable=False)
    processing_started_at = Column("processingStartedAt", DateTime(timezone=True))
    processing_completed_at = Column("processingCompletedAt", DateTime(timezone=True))
    error_message = Column("errorMessage", Text)
    retry_count = Column("retryCount", Integer, default=0)
    created_at = Column("createdAt", DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column("updatedAt", DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    playlist = relationship("Playlist", back_populates="videos")
    transcript = relationship("Transcript", back_populates="video", uselist=False)
    
    # Indexes
    __table_args__ = (
        Index("video_youtube_id_idx", "youtubeId"),
        Index("video_playlist_id_idx", "playlistId"),
        Index("video_transcript_id_idx", "transcriptId"),
        Index("video_status_idx", "status"),
        Index("video_published_at_idx", "publishedAt"),
    )

class DatabaseManager:
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database connection with connection resilience."""
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        # Fix postgres:// to postgresql:// for modern SQLAlchemy
        if self.database_url.startswith('postgres://'):
            self.database_url = self.database_url.replace('postgres://', 'postgresql://', 1)
        
        # Configure engine with connection resilience
        self.engine = create_engine(
            self.database_url,
            # Connection pool settings for resilience
            pool_size=5,  # Number of connections to maintain
            max_overflow=10,  # Additional connections when pool is full
            pool_timeout=30,  # Timeout for getting connection from pool
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_pre_ping=True,  # Test connections before use
            # Connection settings
            connect_args={
                "connect_timeout": 10,  # Connection timeout
                "application_name": "podsearch_pipeline",  # Identify connections
                "keepalives_idle": 30,  # Send keepalive after 30s of inactivity
                "keepalives_interval": 10,  # Send keepalive every 10s
                "keepalives_count": 5,  # Give up after 5 failed keepalives
            }
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a database session with retry logic for connection resilience."""
        return self.SessionLocal()
    
    def execute_with_retry(self, operation, max_retries=3, base_delay=1):
        """
        Execute a database operation with retry logic for connection resilience.
        
        Args:
            operation: Function that takes a session and performs the operation
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (will be exponential)
            
        Returns:
            Result of the operation if successful
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                with self.get_session() as session:
                    result = operation(session)
                    return result
                    
            except (OperationalError, DisconnectionError) as e:
                last_exception = e
                
                # Check if it's an SSL connection error
                if "SSL connection has been closed" in str(e) or "connection" in str(e).lower():
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"⚠️ Database connection error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        print(f"   Retrying in {delay} seconds...")
                        time.sleep(delay)
                        
                        # Force engine to dispose and recreate connections
                        if attempt == max_retries - 1:  # Last retry attempt
                            print("   Recreating database engine...")
                            self.engine.dispose()
                        continue
                    else:
                        print(f"❌ Database connection failed after {max_retries + 1} attempts")
                        raise
                else:
                    # Non-connection related error, don't retry
                    raise
                    
            except Exception as e:
                # Non-connection related error, don't retry
                raise
                
        # This should never be reached, but just in case
        raise last_exception or Exception("Unknown database error")

    def get_or_create_playlist(self, youtube_id: str, title: str, description: str = None, 
                              channel_name: str = None, channel_id: str = None, 
                              url: str = None) -> Playlist:
        """Get existing playlist or create new one."""
        with self.get_session() as session:
            playlist = session.query(Playlist).filter(Playlist.youtube_id == youtube_id).first()
            
            if not playlist:
                playlist = Playlist(
                    youtube_id=youtube_id,
                    title=title,
                    description=description,
                    channel_name=channel_name,
                    channel_id=channel_id,  
                    url=url,
                    created_at=datetime.now()
                )
                session.add(playlist)
                session.commit()
                session.refresh(playlist)
            else:
                # Update existing playlist info
                playlist.title = title
                playlist.description = description  
                playlist.channel_name = channel_name
                playlist.channel_id = channel_id
                playlist.url = url
                playlist.updated_at = datetime.now()
                session.commit()
                
            return playlist
    
    def get_or_create_video(self, youtube_id: str, playlist_id: int, title: str,
                           description: str = None, duration: int = None,
                           published_at: datetime = None, thumbnail_url: str = None,
                           url: str = None) -> Video:
        """Get existing video or create new one."""
        with self.get_session() as session:
            video = session.query(Video).filter(Video.youtube_id == youtube_id).first()
            
            if not video:
                video = Video(
                    youtube_id=youtube_id,
                    playlist_id=playlist_id,
                    title=title,
                    description=description,
                    duration=duration,
                    published_at=published_at,
                    thumbnail_url=thumbnail_url,
                    url=url,
                    status=VideoStatus.PENDING,
                    created_at=datetime.now()
                )
                session.add(video)
                session.commit()
                session.refresh(video)
            
            return video
    
    def update_video_status(self, video_id: int, status: str, error_message: str = None,
                           local_file_path: str = None):
        """Update video processing status."""
        with self.get_session() as session:
            video = session.query(Video).filter(Video.id == video_id).first()
            if video:
                video.status = status
                video.updated_at = datetime.now()
                
                if error_message:
                    video.error_message = error_message
                    video.retry_count += 1
                
                if local_file_path:
                    video.local_file_path = local_file_path
                
                if status == VideoStatus.DOWNLOADED:
                    video.processing_started_at = datetime.now()
                elif status in [VideoStatus.FINISHED, VideoStatus.FAILED]:
                    video.processing_completed_at = datetime.now()
                
                session.commit()
    
    def get_videos_by_status(self, status: str, playlist_id: int = None, limit: int = None) -> List[Video]:
        """Get videos by status, optionally filtered by playlist."""
        with self.get_session() as session:
            query = session.query(Video).filter(Video.status == status)
            if playlist_id:
                query = query.filter(Video.playlist_id == playlist_id)
            query =  query.order_by(Video.published_at.desc())
            if limit:
                query = query.limit(limit)
            return query.all()
    
    def get_pending_videos(self, playlist_id: int = None) -> List[Video]:
        """Get videos ready for processing."""
        return self.get_videos_by_status(VideoStatus.PENDING, playlist_id)
    
    def get_video_by_id(self, video_id: int) -> Optional[Video]:
        """Get a video by its database ID."""
        with self.get_session() as session:
            return session.query(Video).filter(Video.id == video_id).first()
    
    def save_transcript_data(self, video_id: int, transcript_data: TranscriptData) -> bool:
        """
        Save transcript data to database using ORM with connection resilience.
        
        Args:
            video_id: Database ID of the video
            transcript_data: Transcript data to save
            
        Returns:
            True if successful, False otherwise
        """
        def save_operation(session):
            # Get the video
            video = session.query(Video).filter(Video.id == video_id).first()
            if not video:
                print(f"❌ Video with ID {video_id} not found")
                return False
            
            # Check if video already has a transcript
            if video.transcript_id:
                # Update existing transcript
                transcript = session.query(Transcript).filter(Transcript.id == video.transcript_id).first()
                if transcript:
                    transcript.language = transcript_data["language"]
                    transcript.segments = transcript_data["segments"]
                    transcript.processing_metadata = {
                        "processed_at": transcript_data["processed_at"],
                        "segments_count": len(transcript_data["segments"])
                    }
                    transcript.updated_at = datetime.now()
            else:
                # Create new transcript
                transcript = Transcript(
                    language=transcript_data["language"],
                    segments=transcript_data["segments"],
                    processing_metadata={
                        "processed_at": transcript_data["processed_at"],
                        "segments_count": len(transcript_data["segments"])
                    }
                )
                session.add(transcript)
                session.flush()  # Get the transcript ID
                
                # Link transcript to video
                video.transcript_id = transcript.id
            
            # Update video timestamp
            video.updated_at = datetime.now()
            
            session.commit()
            print(f"✅ Transcript saved to database for video {video_id}")
            return True
        
        try:
            return self.execute_with_retry(save_operation)
        except Exception as e:
            print(f"❌ Error saving transcript to database: {e}")
            return False
    
    def update_playlist_sync(self, playlist_id: int, total_videos: int):
        """Update playlist sync information.""" 
        with self.get_session() as session:
            playlist = session.query(Playlist).filter(Playlist.id == playlist_id).first()
            if playlist:
                playlist.total_videos = total_videos
                playlist.last_sync_at = datetime.now()
                playlist.updated_at = datetime.now()
                session.commit() 

    def get_playlist_by_id(self, playlist_id: int) -> Optional[Playlist]:
        """Get a playlist by its database ID."""
        with self.get_session() as session:
            return session.query(Playlist).filter(Playlist.id == playlist_id).first()