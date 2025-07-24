#!/usr/bin/env python3
"""
Database models and connection for the video processing pipeline.
Matches the Drizzle schema in schema.ts
"""

import os
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
from dotenv import load_dotenv

load_dotenv()

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

class Video(Base):
    __tablename__ = "podsearch_video"
    
    id = Column(Integer, primary_key=True, autoincrement=True)  
    youtube_id = Column("youtubeId", String(255), nullable=False, unique=True)
    playlist_id = Column("playlistId", Integer, ForeignKey("podsearch_playlist.id", ondelete="CASCADE"))
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
    
    # Indexes
    __table_args__ = (
        Index("video_youtube_id_idx", "youtubeId"),
        Index("video_playlist_id_idx", "playlistId"),
        Index("video_status_idx", "status"),
        Index("video_published_at_idx", "publishedAt"),
    )

class DatabaseManager:
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database connection."""
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        # Fix postgres:// to postgresql:// for modern SQLAlchemy
        if self.database_url.startswith('postgres://'):
            self.database_url = self.database_url.replace('postgres://', 'postgresql://', 1)
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
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
    
    def get_videos_by_status(self, status: str, playlist_id: int = None) -> List[Video]:
        """Get videos by status, optionally filtered by playlist."""
        with self.get_session() as session:
            query = session.query(Video).filter(Video.status == status)
            if playlist_id:
                query = query.filter(Video.playlist_id == playlist_id)
            return query.order_by(Video.published_at.desc()).all()
    
    def get_pending_videos(self, playlist_id: int = None) -> List[Video]:
        """Get videos ready for processing."""
        return self.get_videos_by_status(VideoStatus.PENDING, playlist_id)
    
    def update_playlist_sync(self, playlist_id: int, total_videos: int):
        """Update playlist sync information.""" 
        with self.get_session() as session:
            playlist = session.query(Playlist).filter(Playlist.id == playlist_id).first()
            if playlist:
                playlist.total_videos = total_videos
                playlist.last_sync_at = datetime.now()
                playlist.updated_at = datetime.now()
                session.commit() 