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
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Index, Boolean, Float, text
from sqlalchemy.dialects.postgresql import insert
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

class ChapterSimilarityData(TypedDict):
    source_chapter_id: int
    dest_chapter_id: int
    similarity_score: float

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
    chapters = relationship("Chapter", back_populates="video", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("video_youtube_id_idx", "youtubeId"),
        Index("video_playlist_id_idx", "playlistId"),
        Index("video_transcript_id_idx", "transcriptId"),
        Index("video_status_idx", "status"),
        Index("video_published_at_idx", "publishedAt"),
    )

class Chapter(Base):
    __tablename__ = "podsearch_chapter"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column("videoId", Integer, ForeignKey("podsearch_video.id", ondelete="CASCADE"), nullable=False)
    chapter_idx = Column("chapterIdx", Integer, nullable=False)  # YouTube chapter index
    chapter_name = Column("chapterName", String(500), nullable=False)
    chapter_summary = Column("chapterSummary", Text, nullable=False)  # LLM-generated summary
    start_time = Column("startTime", Integer, nullable=False)  # start time in seconds
    end_time = Column("endTime", Integer, nullable=False)  # end time in seconds
    created_at = Column("createdAt", DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column("updatedAt", DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    video = relationship("Video", back_populates="chapters")
    source_similarities = relationship("ChapterSimilarity", foreign_keys="ChapterSimilarity.source_chapter_id", back_populates="source_chapter")
    dest_similarities = relationship("ChapterSimilarity", foreign_keys="ChapterSimilarity.dest_chapter_id", back_populates="dest_chapter")
    
    # Indexes
    __table_args__ = (
        Index("chapter_video_id_idx", "videoId"),
        Index("chapter_video_idx_idx", "videoId", "chapterIdx"),  # Unique constraint
    )

class ChapterSimilarity(Base):
    __tablename__ = "podsearch_chapter_similarity"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_chapter_id = Column("sourceChapterId", Integer, ForeignKey("podsearch_chapter.id", ondelete="CASCADE"), nullable=False)
    dest_chapter_id = Column("destChapterId", Integer, ForeignKey("podsearch_chapter.id", ondelete="CASCADE"), nullable=False)
    similarity_score = Column("similarityScore", Float, nullable=False)  # Pinecone similarity score
    created_at = Column("createdAt", DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Relationships
    source_chapter = relationship("Chapter", foreign_keys=[source_chapter_id], back_populates="source_similarities")
    dest_chapter = relationship("Chapter", foreign_keys=[dest_chapter_id], back_populates="dest_similarities")
    
    # Indexes
    __table_args__ = (
        Index("chapter_similarity_source_idx", "sourceChapterId"),
        Index("chapter_similarity_dest_idx", "destChapterId"),
        Index("chapter_similarity_score_idx", "similarityScore"),
        Index("chapter_similarity_unique_idx", "sourceChapterId", "destChapterId", unique=True),
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

    # Chapter-related methods
    def get_finished_videos(self, playlist_id: int = None) -> List[Video]:
        """Get videos with 'finished' status that don't have chapters yet."""
        with self.get_session() as session:
            query = session.query(Video).filter(Video.status == VideoStatus.FINISHED)
            if playlist_id:
                query = query.filter(Video.playlist_id == playlist_id)
            query = query.order_by(Video.published_at.desc())
            return query.all()

    def save_chapter(self, video_id: int, chapter_idx: int, chapter_name: str, 
                    chapter_summary: str, start_time: int, end_time: int) -> Optional[Chapter]:
        """Save a chapter to the database."""
        with self.get_session() as session:
            # Check if chapter already exists
            existing = session.query(Chapter).filter(
                Chapter.video_id == video_id,
                Chapter.chapter_idx == chapter_idx
            ).first()
            
            if existing:
                # Update existing chapter
                existing.chapter_name = chapter_name
                existing.chapter_summary = chapter_summary
                existing.start_time = start_time
                existing.end_time = end_time
                existing.updated_at = datetime.now()
                session.commit()
                return existing
            else:
                # Create new chapter
                chapter = Chapter(
                    video_id=video_id,
                    chapter_idx=chapter_idx,
                    chapter_name=chapter_name,
                    chapter_summary=chapter_summary,
                    start_time=start_time,
                    end_time=end_time
                )
                session.add(chapter)
                session.commit()
                session.refresh(chapter)
                return chapter

    def save_chapter_similarity(self, source_chapter_id: int, dest_chapter_id: int, 
                              similarity_score: float) -> ChapterSimilarity:
        """Save a chapter similarity relationship."""
        with self.get_session() as session:
            # Check if similarity already exists
            existing = session.query(ChapterSimilarity).filter(
                ChapterSimilarity.source_chapter_id == source_chapter_id,
                ChapterSimilarity.dest_chapter_id == dest_chapter_id
            ).first()
            
            if existing:
                # Update existing similarity
                existing.similarity_score = similarity_score
                session.commit()
                return existing
            else:
                # Create new similarity
                similarity = ChapterSimilarity(
                    source_chapter_id=source_chapter_id,
                    dest_chapter_id=dest_chapter_id,
                    similarity_score=similarity_score
                )
                session.add(similarity)
                session.commit()
                session.refresh(similarity)
                return similarity

    def bulk_upsert_chapter_similarities(self, similarities: List[ChapterSimilarityData]) -> int:
        """
        Bulk upsert chapter similarities using SQLAlchemy ORM with type safety.
        Much more efficient than individual queries.
        
        Args:
            similarities: List of dicts with keys: source_chapter_id, dest_chapter_id, similarity_score
            
        Returns:
            Number of rows affected
        """
        if not similarities:
            return 0
            
        with self.get_session() as session:
            # Prepare data for bulk insert using SQLAlchemy model column names
            values_to_insert = []
            for sim in similarities:
                values_to_insert.append({
                    ChapterSimilarity.source_chapter_id.key: sim['source_chapter_id'],
                    ChapterSimilarity.dest_chapter_id.key: sim['dest_chapter_id'], 
                    ChapterSimilarity.similarity_score.key: sim['similarity_score']
                })
            
            # Use SQLAlchemy's PostgreSQL insert with on_conflict_do_update
            stmt = insert(ChapterSimilarity).values(values_to_insert)
            stmt = stmt.on_conflict_do_update(
                constraint='chapter_similarity_unique',
                set_={
                    ChapterSimilarity.similarity_score: stmt.excluded.similarityScore,
                    ChapterSimilarity.created_at: func.now()
                }
            )
            
            result = session.execute(stmt)
            session.commit()
            return result.rowcount

    def get_chapters_by_video_id(self, video_id: int) -> List[Chapter]:
        """Get all chapters for a video."""
        with self.get_session() as session:
            return session.query(Chapter).filter(
                Chapter.video_id == video_id
            ).order_by(Chapter.chapter_idx).all()

    def get_chapter_similarities(self, chapter_id: int, limit: int = 5) -> List[ChapterSimilarity]:
        """Get top similar chapters for a given chapter."""
        with self.get_session() as session:
            return session.query(ChapterSimilarity).filter(
                ChapterSimilarity.source_chapter_id == chapter_id
            ).order_by(ChapterSimilarity.similarity_score.desc()).limit(limit).all()

    def get_all_chapters_for_embedding(self) -> List[Chapter]:
        """Get all chapters that need to be embedded (for similarity search)."""
        with self.get_session() as session:
            return session.query(Chapter).all()

    def clear_chapter_similarities_for_video(self, video_id: int):
        """Clear all similarity relationships for chapters in a video."""
        with self.get_session() as session:
            # Get all chapter IDs for this video
            chapter_ids = session.query(Chapter.id).filter(Chapter.video_id == video_id).all()
            chapter_ids = [c[0] for c in chapter_ids]
            
            if chapter_ids:
                # Delete similarities where source or dest is in this video
                session.query(ChapterSimilarity).filter(
                    (ChapterSimilarity.source_chapter_id.in_(chapter_ids)) |
                    (ChapterSimilarity.dest_chapter_id.in_(chapter_ids))
                ).delete()
                session.commit()
    
    def get_all_similarities(self) -> List[ChapterSimilarity]:
        """Get all chapter similarities."""
        with self.get_session() as session:
            return session.query(ChapterSimilarity).limit(100).all()
    
    def get_chapters_by_ids(self, chapter_ids: List[int]) -> List[Chapter]:
        """Get chapters by their database IDs."""
        with self.get_session() as session:
            return session.query(Chapter).filter(Chapter.id.in_(chapter_ids)).all()
    
    def get_videos_with_chapters(self) -> List[Video]:
        """Get all videos that have chapters."""
        with self.get_session() as session:
            # Use a subquery to find videos that have chapters
            videos_with_chapters = session.query(Video).join(Chapter).distinct().all()
            return videos_with_chapters