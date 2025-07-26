# Diary of a CEO Search Engine Setup

## Overview

This is an AI-powered search engine for the "Diary of a CEO" podcast that allows users to:

1. Search for topics using natural language
2. Receive AI-generated clarifying questions to refine their search
3. Get 5 personalized video clips that jump directly to relevant moments
4. Watch clips with timestamps and transcripts

## Architecture

- **Frontend**: Next.js with React YouTube player and Tailwind CSS
- **Backend**: tRPC with TypeScript for type-safe APIs
- **Database**: PostgreSQL with Drizzle ORM
- **Search**: Pinecone vector database for semantic search
- **AI**: OpenAI GPT-4 for clarifying questions and OpenAI embeddings for search
- **Audio Processing**: Python scripts with Whisper for transcript extraction

## Required Environment Variables

Create a `.env` file with:

```bash
# Database
DATABASE_URL="postgresql://username:password@localhost:5432/database_name"

# Authentication (Discord OAuth)
AUTH_SECRET="your-nextauth-secret-here"
AUTH_GOOGLE_ID="your-discord-client-id"
AUTH_GOOGLE_SECRET="your-discord-client-secret"

# AI APIs
OPENAI_API_KEY="your-openai-api-key"
PINECONE_API_KEY="your-pinecone-api-key"
```

## Setup Steps

### 1. Install Dependencies

```bash
npm install
```

### 2. Database Setup

```bash
# Generate migration files
npm run db:generate

# Apply migrations to database
npm run db:migrate
```

### 3. Pinecone Setup

- Create a Pinecone index named `video-segments`
- Use 3072 dimensions (for OpenAI text-embedding-3-large)
- Use cosine similarity metric

### 4. Process Videos

Update your video processing pipeline to include YouTube IDs:

```python
# In your process_video.py script, ensure you're storing:
metadata = {
    "video_name": str(segment["video_name"]),
    "video_url": str(segment["video_url"]),
    "youtube_id": str(segment.get("youtube_id", "")),  # <- This is crucial
    "segment_id": int(segment["segment_id"]),
    "start_time": float(segment["start_time"]),
    "end_time": float(segment["end_time"]),
    "duration": float(segment["duration"]),
    "timestamp_readable": str(segment["timestamp_readable"]),
    "full_text_length": int(len(segment["text"]))
}
```

### 5. Populate Database

Store your transcript data in the database:

```sql
-- Example transcript insert
INSERT INTO podsearch_transcript (
    video_id, youtube_id, full_text, segments,
    language, duration, segments_count, whisper_model
) VALUES (
    1, 'dQw4w9WgXcQ', 'Full transcript text...',
    '[{"id": 1, "text": "segment text", "start": 0, "end": 30}]'::jsonb,
    'en', 1200.5, 40, 'base'
);
```

## Usage Flow

1. **Search Input**: User enters a topic they're interested in
2. **Broad Search**: System searches Pinecone for 15 relevant segments
3. **AI Clarification**: GPT-4 generates 3-4 clarifying questions based on results
4. **User Responses**: User answers clarifying questions
5. **Refined Search**: AI selects the top 5 most relevant segments
6. **Results Display**: YouTube players with timestamps + transcripts

## API Endpoints

### tRPC Routes

- `search.broadSearch({ query, topK })` - Initial semantic search
- `search.generateClarifyingQuestions({ originalQuery, searchResults })` - Generate follow-up questions
- `search.refinedSearch({ originalQuery, clarifyingAnswers, originalResults })` - Refined results
- `search.getSegmentTranscript({ youtubeId, segmentId })` - Get full transcript for a segment

## File Structure

```
src/
├── pages/
│   ├── index.tsx          # Landing page
│   └── search.tsx         # Main search interface
├── server/
│   ├── api/
│   │   ├── routers/
│   │   │   └── search.ts  # tRPC search router
│   │   └── root.ts        # Main tRPC router
│   └── db/
│       └── schema.ts      # Database schema
└── utils/
    └── api.ts             # tRPC client setup

scripts/
└── process_video.py       # Updated with YouTube ID support
```

## Development

```bash
# Start development server
npm run dev

# Database management
npm run db:studio    # Open Drizzle Studio
npm run db:push      # Push schema changes
```

## Production Considerations

1. **Rate Limiting**: Add rate limiting for OpenAI API calls
2. **Caching**: Cache search results and clarifying questions
3. **Error Handling**: Robust error handling for AI failures
4. **Performance**: Consider pagination for large result sets
5. **Security**: Validate all user inputs and sanitize data

## Troubleshooting

### Common Issues

1. **Pinecone Connection**: Ensure index name matches (`video-segments`)
2. **YouTube ID Missing**: Make sure process_video.py includes YouTube IDs in metadata
3. **Database Connection**: Verify DATABASE_URL format
4. **OpenAI Limits**: Check API usage and rate limits

### Debug Mode

Set environment variable for more verbose logging:

```bash
SKIP_ENV_VALIDATION=true
```

## Next Steps

1. Add user authentication and search history
2. Implement search filters (date, emotions, guests)
3. Add social features (sharing, favorites)
4. Implement better error handling and loading states
5. Add analytics and search metrics
