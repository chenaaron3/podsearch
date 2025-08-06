# Chapter Processing Pipeline

This pipeline processes YouTube chapters from finished videos to create a knowledge graph of related content.

## Overview

The chapter pipeline:

1. **Extracts chapters** from YouTube videos using the YouTube API
2. **Generates summaries** using OpenAI GPT-4 to distill each chapter into one main concept
3. **Creates embeddings** for chapter summaries and stores them in Pinecone
4. **Finds similar chapters** across different videos using vector similarity
5. **Stores relationships** in the database for graph visualization

## Database Schema

### New Tables

#### `podsearch_chapter`

- `id`: Primary key
- `videoId`: Reference to video
- `chapterIdx`: YouTube chapter index (0, 1, 2...)
- `chapterName`: Chapter title from YouTube
- `chapterSummary`: LLM-generated summary (1-2 sentences)
- `startTime`: Chapter start time in seconds
- `endTime`: Chapter end time in seconds

#### `podsearch_chapter_similarity`

- `id`: Primary key
- `sourceChapterId`: Source chapter for similarity
- `destChapterId`: Destination chapter (similar to source)
- `similarityScore`: Pinecone similarity score (0-1)

## Usage

### Prerequisites

1. **Environment Variables**:

   ```bash
   DATABASE_URL=postgresql://...
   OPENAI_API_KEY=sk-...
   PINECONE_API_KEY=...
   ```

2. **Database Migration**:
   ```bash
   npm run db:push
   ```

### Processing Videos

#### Process all finished videos:

```bash
python scripts/chapter_pipeline.py --all-finished
```

#### Process videos in a specific playlist:

```bash
python scripts/chapter_pipeline.py --playlist-id 123
```

### Testing

Run the test suite to verify everything is working:

```bash
python scripts/test_chapter_pipeline.py
```

## API Endpoints

The pipeline creates new tRPC endpoints for accessing chapter data:

### Get chapters for a video:

```typescript
const chapters = await trpc.chapters.getChaptersByVideo.query({ videoId: 123 });
```

### Get similar chapters:

```typescript
const similar = await trpc.chapters.getSimilarChapters.query({
  chapterId: 456,
  limit: 5,
});
```

### Get graph data for visualization:

```typescript
const graphData = await trpc.chapters.getGraphData.query({
  videoId: 123, // optional
  limit: 50,
});
```

### Get statistics:

```typescript
const stats = await trpc.chapters.getStats.query();
```

## Data Flow

1. **Video Processing**: Videos must be in "finished" status (have transcripts)
2. **Chapter Extraction**: YouTube chapters are fetched using existing `hook_finder.py`
3. **Text Extraction**: Transcript segments within chapter timeframes are extracted
4. **Summarization**: OpenAI GPT-4 generates 1-2 sentence summaries using `summarize.txt` prompt
5. **Embedding**: Chapter summaries are embedded using OpenAI text-embedding-ada-002
6. **Similarity Search**: Pinecone finds top 5 similar chapters from different videos
7. **Relationship Storage**: Similarities are stored as edges in the database

## Key Features

- **Skip Processing**: Videos that already have chapters are skipped
- **Cross-Video Similarity**: Chapters only link to chapters from different videos
- **One-Directional Edges**: Each similarity is stored once (A→B, not B→A)
- **Batch Processing**: Processes multiple videos efficiently
- **Error Handling**: Robust error handling with retry logic
- **Status Tracking**: Clear progress reporting

## Graph Visualization

The API provides graph data in a format suitable for visualization libraries:

```typescript
interface GraphData {
  nodes: GraphNode[]; // Chapters
  edges: GraphEdge[]; // Similarities
}
```

Each node represents a chapter with metadata, and each edge represents a similarity relationship with a score.

## Monitoring

Check processing status:

```bash
# Check database for processed chapters
python -c "
from database import DatabaseManager
db = DatabaseManager()
chapters = db.get_all_chapters_for_embedding()
print(f'Total chapters: {len(chapters)}')
"
```

## Troubleshooting

### Common Issues

1. **No chapters found**: Some videos don't have YouTube chapters
2. **OpenAI API errors**: Check API key and rate limits
3. **Pinecone errors**: Verify index exists and API key is valid
4. **Database errors**: Ensure migration has been run

### Debug Mode

Run with verbose logging:

```bash
python scripts/chapter_pipeline.py --all-finished 2>&1 | tee chapter_processing.log
```

## Future Enhancements

- **Incremental Updates**: Only process new videos
- **Similarity Thresholds**: Filter low-quality similarities
- **Bidirectional Edges**: Store both directions for easier querying
- **Chapter Clustering**: Group similar chapters into topics
- **Real-time Processing**: Process chapters as videos are finished
