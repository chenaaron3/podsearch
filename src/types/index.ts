export type SearchSegment = {
  id: string;
  score: number;
  youtubeId: string;
  videoTitle: string;
  segmentId: number;
  startTime: number;
  endTime: number;
  duration: number;
  timestampReadable: string;
  transcriptText: string;
  primaryEmotion?: string;
  emotionScore?: number;
};

// Transcript segment types
export interface TranscriptWord {
  word: string;
  start: number;
  end: number;
}

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  words?: TranscriptWord[];
}

export interface TranscriptData {
  segments: TranscriptSegment[];
}

// Pinecone search query types
export interface PineconeSearchQuery {
  vector: number[];
  topK: number;
  includeMetadata: boolean;
  filter?: {
    video_id: { $eq: number };
  };
}

// Search response types
export interface SearchResponse {
  segments: SearchSegment[];
  totalFound: number;
  query: string;
  videoInfo?: {
    id: number;
    youtubeId: string;
    title: string;
  };
}

// Video details type
export interface VideoDetails {
  youtubeId: string;
  title: string;
  id: number;
  transcriptId: number | null;
  transcriptSegments: TranscriptSegment[] | null;
}

// Word timestamp type
export interface WordTimestamp {
  word: string;
  start: number;
  end: number;
}

// YouTube Player types
export interface YouTubePlayer {
  // Player control methods
  playVideo(): void;
  pauseVideo(): void;
  stopVideo(): void;
  seekTo(seconds: number, allowSeekAhead?: boolean): void;
  loadVideoById(videoId: string, startSeconds?: number): void;
  cueVideoById(videoId: string, startSeconds?: number): void;

  // Player state methods
  getPlayerState(): number;
  getCurrentTime(): number;
  getDuration(): number;
  getVideoLoadedFraction(): number;
  getPlayerResponse(): unknown;
  getVideoUrl(): string;
  getVideoData(): unknown;

  // Player settings methods
  mute(): void;
  unMute(): void;
  isMuted(): boolean;
  setVolume(volume: number): void;
  getVolume(): number;
  setPlaybackRate(suggestedRate: number): void;
  getPlaybackRate(): number;
  getAvailablePlaybackRates(): number[];
  setLoop(loopPlaylists: boolean): void;
  setShuffle(shufflePlaylist: boolean): void;

  // Player information methods
  getVideoBytesLoaded(): number;
  getVideoBytesTotal(): number;
  getPlaylist(): string[];
  getPlaylistIndex(): number;

  // Player event methods
  addEventListener(event: string, listener: (event: unknown) => void): void;
  removeEventListener(event: string, listener: (event: unknown) => void): void;

  // Player element methods
  getIframe(): HTMLIFrameElement;
  destroy(): void;
}

// YouTube event types
export interface YouTubeEvent {
  target: YouTubePlayer;
  data?: number | string;
}

// YouTube refs type
export type YouTubeRefs = Record<string, YouTubePlayer>;
