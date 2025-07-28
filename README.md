# PodSearch - AI-Powered Podcast Search

A sophisticated search engine for podcast episodes that uses AI to find and extract the most relevant moments from audio content.

## Features

### 🔍 Smart Search

- **Global Search**: Search across all podcast episodes to find relevant segments
- **Episode-Specific Search**: Search within a specific episode to find popular moments and insights
- **AI-Powered Ranking**: Uses GPT-4 to intelligently rank and select the most relevant clips
- **Precise Timestamping**: Finds exact starting points for answers within segments
- **Emotion Analysis**: Identifies emotional context of segments

### 🎯 Search Within Video

- **Follow-up Queries**: After finding an interesting episode, search for specific topics within that episode
- **Popular Moments**: Discover the most relevant and engaging moments from a particular episode
- **Contextual Search**: Get more targeted results when you know which episode contains what you're looking for

### 🎬 Video Processing Pipeline

- **Automatic Transcription**: Uses Whisper for high-quality speech-to-text
- **Semantic Segmentation**: Breaks episodes into meaningful segments based on topic changes
- **Vector Embeddings**: Stores semantic representations for fast similarity search
- **Pinecone Integration**: Scalable vector database for efficient retrieval

## Architecture

### Backend

- **Next.js API Routes**: RESTful endpoints for search functionality
- **tRPC**: Type-safe API layer
- **Drizzle ORM**: Database operations with PostgreSQL
- **OpenAI API**: Embeddings and LLM-powered ranking
- **Pinecone**: Vector database for similarity search

### Frontend

- **React**: Modern UI with TypeScript
- **Tailwind CSS**: Styled components and responsive design
- **YouTube Player**: Embedded video playback with precise timestamping
- **Mobile-First**: Touch-friendly interface with swipe gestures

## Search Modes

### 1. Global Search

Search across all available podcast episodes to find the most relevant content for your query.

**Use Cases:**

- Finding specific advice or insights across multiple episodes
- Discovering episodes that cover particular topics
- Getting a broad overview of content related to your interests

### 2. Episode-Specific Search

Search within a specific episode to find the most relevant moments and insights.

**Use Cases:**

- Finding specific moments in an episode you're interested in
- Discovering the most popular or important segments of an episode
- Getting more targeted results when you know which episode contains what you're looking for

**How to Use:**

1. Perform a global search to find relevant episodes
2. Click "Search this episode" on any result to switch to episode-specific search
3. Enter your follow-up query to find specific moments within that episode

## API Endpoints

### `GET /api/trpc/search.search`

Unified search endpoint that handles both global and video-specific searches.

**Parameters:**

- `query`: Search query string
- `topK`: Number of results to return (default: 5)
- `videoId`: (Optional) Database ID of the video to search within. If provided, search is limited to that video.

**Examples:**

```javascript
// Global search across all episodes
await api.search.search.fetch({
  query: "startup advice",
  topK: 5,
});

// Search within a specific video
await api.search.search.fetch({
  query: "leadership tips",
  videoId: 123,
  topK: 10,
});
```

### `GET /api/trpc/search.getVideoByYoutubeId`

Get video details by YouTube ID.

**Parameters:**

- `youtubeId`: YouTube video ID

## Setup

1. **Install Dependencies**

   ```bash
   npm install
   ```

2. **Environment Variables**
   Create a `.env.local` file with:

   ```
   DATABASE_URL=your_postgresql_connection_string
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

3. **Database Setup**

   ```bash
   npm run db:push
   ```

4. **Start Development Server**
   ```bash
   npm run dev
   ```

## Processing Pipeline

1. **Video Download**: Downloads videos from YouTube playlists
2. **Transcription**: Uses Whisper to generate word-level timestamps
3. **Segmentation**: Creates semantic segments based on topic changes
4. **Emotion Analysis**: Analyzes emotional context of segments
5. **Embedding Generation**: Creates vector embeddings for similarity search
6. **Vector Storage**: Stores embeddings in Pinecone for fast retrieval

## Search Algorithm

1. **Query Embedding**: Converts search query to vector representation
2. **Vector Search**: Finds similar segments using cosine similarity
3. **Database Filtering**: Retrieves full transcript data for timestamp-based filtering
4. **LLM Ranking**: Uses GPT-4 to rank segments by relevance
5. **Precise Seeking**: Finds exact starting points within segments
6. **Result Formatting**: Returns structured results with timestamps and transcripts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
