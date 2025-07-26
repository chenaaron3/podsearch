# Video Processing Pipeline

Complete pipeline for processing YouTube videos with AI-powered transcript extraction, semantic segmentation, emotion analysis, and vector embeddings.

## Features

- **Database-Driven Processing**: Uses PostgreSQL database as source of truth for video metadata
- **Whisper Transcript Extraction**: High-quality transcript extraction with timestamps
- **Semantic Segmentation**: Intelligent grouping into 1-2 minute segments based on topic changes
- **Emotion Analysis**: GoEmotions-based emotion detection for each segment (28 emotion categories)
- **Vector Embeddings**: OpenAI or local sentence transformer embeddings for similarity search
- **Pinecone Integration**: Stores embeddings in Pinecone for fast semantic search
- **Batch Processing**: Process individual videos or entire playlists

## Architecture

```
YouTube Playlist/Channel
         ↓
    Playlist Fetcher (saves to DB)
         ↓
    Video Downloader
         ↓
    Video Processor (NEW: DB-driven)
         ↓
    Pinecone Vector Store
```

## Database Integration

The video processor now uses the database as the source of truth instead of extracting metadata from filenames:

- Fetches video metadata (title, YouTube ID, URL, file path) from `podsearch_video` table
- Saves transcript data to `podsearch_transcript` table
- Updates video processing status in the database
- Links Pinecone vectors back to database records via YouTube ID

## Setup

1. **Install Dependencies**

   ```bash
   cd scripts
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file with:

   ```bash
   DATABASE_URL=postgresql://user:password@localhost:5432/podsearch
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

3. **Database Setup**
   Ensure your PostgreSQL database is running with the correct schema (see `../src/server/db/schema.ts`)

## Usage

### Process a Single Video

```bash
python process_video.py --video-id 123
```

### Batch Process All Pending Videos

```bash
python process_video.py --batch-process
```

### Process Videos from Specific Playlist

```bash
python process_video.py --batch-process --playlist-id 456
```

### Force Reprocessing (Ignore Cache)

```bash
python process_video.py --video-id 123 --force-reprocess
```

### Use Local Embeddings (Instead of OpenAI)

```bash
python process_video.py --batch-process --embedding-type local
```

### Disable Emotion Analysis

```bash
python process_video.py --batch-process --disable-emotions
```

## Complete Pipeline Workflow

1. **Sync Playlist**: Fetch video metadata from YouTube and save to database

   ```bash
   python playlist_pipeline.py "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
   ```

2. **Download & Process**: The pipeline automatically downloads and processes videos
   - Downloads videos to local storage
   - Processes transcripts and generates embeddings
   - Updates video status in database

3. **Search**: Use the web interface or search script
   ```bash
   python search_segments.py "your search query"
   ```

## Configuration Options

### Embedding Types

- **OpenAI** (default): Uses `text-embedding-3-large` (3072 dimensions)
- **Local**: Uses sentence transformers (configurable model)

### Emotion Analysis

- **Enabled** (default): Uses GoEmotions model for 28 emotion categories
- **Disabled**: Skips emotion analysis for faster processing

### Processing Parameters

- **Target segment duration**: 45 seconds
- **Minimum segment duration**: 30 seconds
- **Maximum segment duration**: 60 seconds
- **Similarity threshold**: 0.85 for grouping similar segments

## Output Files

All processed data is saved in organized directories:

```
processed/
├── transcripts/          # Full transcripts with timestamps
│   └── {youtube_id}_transcript.json
├── segments/            # Semantic segments
│   └── {youtube_id}_segments.json
├── embeddings/          # Vector embeddings
│   └── {youtube_id}_embeddings.json
├── emotions/            # Emotion analysis results
│   └── {youtube_id}_emotions.json
└── {video_id}_analysis.json    # Processing summary
```

## Database Schema

### Videos Table (`podsearch_video`)

- Video metadata, processing status, file paths
- Links to playlist and transcript data

### Transcripts Table (`podsearch_transcript`)

- Full transcript text and segmented data
- Processing metadata and timestamps
- One-to-one relationship with videos

### Pinecone Metadata

Each vector in Pinecone includes:

- `youtube_id`: Links back to database
- `segment_id`: Segment number within video
- `start_time`, `end_time`: Timestamp boundaries
- `primary_emotion`: Dominant emotion if enabled
- Video title and URL for display

## Error Handling

- Database connection errors are logged and gracefully handled
- Failed video processing updates status in database
- Retry mechanisms for transient failures
- Comprehensive error logging for debugging

## Performance

- **Batch processing**: Processes multiple videos sequentially
- **Embedding batching**: Handles API rate limits efficiently
- **Resume capability**: Can resume from any point in the pipeline
- **Status tracking**: Database tracks processing state for each video

## Troubleshooting

1. **Database Connection Issues**
   - Verify `DATABASE_URL` in `.env`
   - Ensure PostgreSQL is running
   - Check database permissions

2. **API Key Issues**
   - Verify `OPENAI_API_KEY` and `PINECONE_API_KEY`
   - Check API quotas and limits

3. **File Not Found Errors**
   - Ensure video files exist at `localFilePath` in database
   - Check file permissions and storage availability

4. **Processing Failures**
   - Check video status in database (`status` column)
   - Review `errorMessage` field for specific errors
   - Use `--force-reprocess` to retry failed videos
