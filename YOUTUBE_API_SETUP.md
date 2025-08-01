# YouTube API Setup for Transcript API

## Current Implementation Status

The API endpoint is now using the official YouTube Data API v3 to:

1. ✅ Verify videos exist
2. ✅ Check if captions are available
3. ✅ List available caption tracks
4. ❌ Get actual caption content (requires OAuth2)

## Options for Getting Actual Transcript Content

### Option 1: OAuth2 Implementation (Recommended for Production)

To get actual caption content, you need to implement OAuth2 authentication:

1. **Set up OAuth2 credentials** in Google Cloud Console
2. **Implement OAuth2 flow** for user authentication
3. **Use the `captions.download` endpoint** with proper authentication

### Option 2: Hybrid Approach (Current Implementation)

The current implementation:

- Uses YouTube API to verify video exists and has captions
- Returns a placeholder message indicating OAuth2 is needed
- Logs all requests to the database

### Option 3: Third-party Services

Consider using services like:

- **YouTube Transcript API** (the package we tried earlier)
- **Whisper API** for audio transcription
- **AssemblyAI** or similar services

## Environment Setup

Add to your `.env` file:

```
YOUTUBE_API_KEY=your_api_key_here
```

## Getting a YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the YouTube Data API v3
4. Create credentials (API Key)
5. Add the key to your `.env` file

## Next Steps

1. **For immediate use**: The current API works and verifies videos
2. **For full functionality**: Implement OAuth2 flow for caption download
3. **For simplicity**: Consider using a third-party transcript service

## API Endpoint

The endpoint is available at:

```
GET /api/transcript?youtubeId=VIDEO_ID&timestamp=SECONDS&duration=SECONDS
```

Returns:

```json
{
  "transcript": "string"
}
```

## Database Logging

All requests are logged to the `transcriptRequests` table with:

- YouTube ID, timestamp, duration
- Success/failure status
- Processing time
- Error messages (if any)
