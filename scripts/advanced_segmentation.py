#!/usr/bin/env python3
"""
Advanced Topic-Based Video Segmentation

Uses embedding-based change point detection and hierarchical topic modeling
to create semantically coherent video segments.

Features:
- Sliding window embeddings for fine-grained analysis
- Change point detection for topic boundary identification
- BERTopic for hierarchical topic modeling
- Multi-scale segmentation (sentence, paragraph, topic levels)
- Visualization of topic evolution and boundaries

Usage:
    python advanced_segmentation.py
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Advanced libraries
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
import ruptures as rpt  # Change point detection
from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx

# For transcript processing
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    required_resources = ['punkt', 'punkt_tab']
    
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {resource}: {e}")
    
    # Fallback: try to download punkt if tokenization fails
    try:
        sent_tokenize("Test sentence.")
    except Exception:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            print(f"Warning: NLTK tokenizer setup failed: {e}")

# Initialize NLTK
ensure_nltk_data()

class AdvancedTopicSegmenter:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 window_size: int = 45,  # seconds - increased for stability
                 overlap_ratio: float = 0.25,  # reduced overlap to reduce noise
                 min_segment_duration: int = 90,  # increased minimum
                 max_segment_duration: int = 240):
        """
        Initialize advanced topic-based segmentation system.
        
        Args:
            embedding_model: Sentence transformer model name
            window_size: Size of sliding window in seconds
            overlap_ratio: Overlap between windows (0.0-1.0)
            min_segment_duration: Minimum segment length in seconds
            max_segment_duration: Maximum segment length in seconds
        """
        self.embedding_model_name = embedding_model  # Store the model name
        self.embedding_model = SentenceTransformer(embedding_model)
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        
        # Initialize BERTopic with optimized settings
        self.topic_model = None
        self._setup_topic_model()
        
        # Results storage
        self.embeddings = None
        self.change_points = None
        self.segments = None
        self.topics = None
        
    def _setup_topic_model(self):
        """Set up BERTopic with optimal configuration for video transcripts."""
        # Reduce dimensionality with UMAP
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # Cluster with HDBSCAN
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=3,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Custom vectorizer for better topic representation
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_features=1000
        )
        
        # Initialize BERTopic
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            top_n_words=10,
            verbose=True
        )

    def create_sliding_windows(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create sliding windows from transcript with semantic boundaries.
        
        Args:
            transcript_data: Transcript with timestamps from Whisper
            
        Returns:
            List of windowed segments with metadata
        """
        print("🪟 Creating sliding windows from transcript...")
        
        segments = transcript_data.get("segments", [])
        windows = []
        
        if not segments:
            print("⚠️  No segments found in transcript data")
            return windows
        
        print(f"📊 Found {len(segments)} transcript segments")
        print(f"🕐 Duration: {transcript_data.get('duration', 'Unknown')} seconds")
        
        # Create sentence-level breakdown first
        sentences = []
        total_text_length = 0
        
        for segment in segments:
            segment_text = segment.get("text", "").strip()
            if not segment_text:
                continue
                
            total_text_length += len(segment_text)
            
            # Split segment text into sentences
            try:
                segment_sentences = sent_tokenize(segment_text)
            except Exception as e:
                print(f"⚠️  Sentence tokenization failed for segment {segment.get('id', 'unknown')}: {e}")
                # Fallback: treat whole segment as one sentence
                segment_sentences = [segment_text]
            
            for i, sentence in enumerate(segment_sentences):
                sentence = sentence.strip()
                if len(sentence) < 5:  # Skip very short sentences
                    continue
                    
                # Estimate sentence timing within segment
                segment_start = segment.get("start", 0)
                segment_end = segment.get("end", segment_start + 1)
                sentence_duration = (segment_end - segment_start) / len(segment_sentences)
                sentence_start = segment_start + (i * sentence_duration)
                sentence_end = sentence_start + sentence_duration
                
                sentences.append({
                    "text": sentence,
                    "start": sentence_start,
                    "end": sentence_end,
                    "segment_id": segment.get("id", i)
                })
        
        print(f"📝 Extracted {len(sentences)} sentences from transcript")
        print(f"📏 Total text length: {total_text_length} characters")
        
        if not sentences:
            print("⚠️  No sentences extracted from transcript")
            return windows
        
        # Create sliding windows with overlap
        step_size = self.window_size * (1 - self.overlap_ratio)
        current_time = 0
        window_id = 0
        
        # Determine total duration from sentences if not available
        total_duration = transcript_data.get("duration", 0)
        if total_duration <= 0 and sentences:
            total_duration = max(s["end"] for s in sentences)
        
        if total_duration <= 0:
            print("⚠️  No valid duration found, creating windows based on sentence count")
            # Fallback: create windows based on sentence groups
            sentences_per_window = max(3, len(sentences) // 10)  # At least 3 sentences per window
            
            for i in range(0, len(sentences), sentences_per_window):
                window_sentences = sentences[i:i + sentences_per_window]
                if window_sentences:
                    window_text = " ".join([s["text"] for s in window_sentences])
                    
                    windows.append({
                        "window_id": window_id,
                        "start_time": window_sentences[0]["start"],
                        "end_time": window_sentences[-1]["end"],
                        "text": window_text,
                        "sentences": window_sentences,
                        "sentence_count": len(window_sentences)
                    })
                    window_id += 1
        else:
            # Normal time-based windowing
            while current_time < total_duration:
                window_end = current_time + self.window_size
                
                # Collect sentences in this window
                window_sentences = []
                for sentence in sentences:
                    # Include sentence if it overlaps with window
                    if (sentence["start"] < window_end and sentence["end"] > current_time):
                        window_sentences.append(sentence)
                
                if window_sentences:
                    window_text = " ".join([s["text"] for s in window_sentences])
                    
                    windows.append({
                        "window_id": window_id,
                        "start_time": current_time,
                        "end_time": min(window_end, total_duration),
                        "text": window_text,
                        "sentences": window_sentences,
                        "sentence_count": len(window_sentences)
                    })
                    window_id += 1
                
                current_time += step_size
        
        print(f"✅ Created {len(windows)} sliding windows")
        return windows

    def generate_embeddings(self, windows: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for all windows.
        
        Args:
            windows: List of windowed segments
            
        Returns:
            Embedding matrix (n_windows, embedding_dim)
        """
        print("🧮 Generating embeddings for windows...")
        
        if not windows:
            print("⚠️  No windows to generate embeddings for")
            return np.array([])
        
        texts = [window["text"] for window in windows if window.get("text", "").strip()]
        
        if not texts:
            print("⚠️  No valid text found in windows")
            return np.array([])
        
        print(f"📊 Generating embeddings for {len(texts)} text segments...")
        
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            self.embeddings = embeddings
            print(f"✅ Generated embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return np.array([])

    def detect_topic_boundaries(self, embeddings: np.ndarray, windows: List[Dict[str, Any]]) -> List[int]:
        """
        Detect topic change points using embedding similarity.
        
        Args:
            embeddings: Window embeddings
            windows: Window metadata
            
        Returns:
            List of change point indices
        """
        print("🔍 Detecting topic boundaries...")
        
        if len(embeddings) == 0:
            print("⚠️  No embeddings available for boundary detection")
            return []
        
        if len(embeddings) < 2:
            print("⚠️  Need at least 2 embeddings for boundary detection")
            return []
        
        print(f"📊 Analyzing {len(embeddings)} embeddings for topic boundaries...")
        
        # Calculate cosine similarity between adjacent windows
        similarities = []
        for i in range(len(embeddings) - 1):
            try:
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[i + 1].reshape(1, -1)
                )[0, 0]
                similarities.append(sim)
            except Exception as e:
                print(f"⚠️  Error calculating similarity at index {i}: {e}")
                similarities.append(0.5)  # Default neutral similarity
        
        if not similarities:
            print("⚠️  No similarities calculated")
            return []
        
        similarities = np.array(similarities)
        
        # Use change point detection on similarity scores
        # Lower similarity = potential topic change
        dissimilarity_scores = 1 - similarities
        
        print(f"📈 Similarity range: {similarities.min():.3f} - {similarities.max():.3f}")
        print(f"📉 Dissimilarity range: {dissimilarity_scores.min():.3f} - {dissimilarity_scores.max():.3f}")
        
        # Apply change point detection with less sensitivity
        try:
            model = "rbf"  # Radial basis function kernel
            # Increased min_size to enforce minimum segment length in windows
            min_windows_per_segment = max(2, self.min_segment_duration // (self.window_size * (1 - self.overlap_ratio)))
            algo = rpt.Pelt(model=model, min_size=min_windows_per_segment, jump=1).fit(dissimilarity_scores.reshape(-1, 1))
            
            # Detect change points with higher penalty for less sensitivity
            change_points = algo.predict(pen=1.5)  # Increased penalty to reduce noise
            
        except Exception as e:
            print(f"⚠️  Change point detection failed: {e}")
            print("🔄 Falling back to simple threshold-based detection...")
            
            # Fallback: simple threshold-based detection (less sensitive)
            threshold = np.percentile(dissimilarity_scores, 85)  # Top 15% as change points (was 75%)
            change_points = []
            for i, score in enumerate(dissimilarity_scores):
                if score > threshold:
                    change_points.append(i + 1)  # +1 because we're looking at transitions
            
            if not change_points:
                # If no change points found, create fewer segments based on duration
                target_segment_duration = (self.min_segment_duration + self.max_segment_duration) / 2
                estimated_total_duration = windows[-1]["end_time"] - windows[0]["start_time"]
                n_segments = max(1, int(estimated_total_duration / target_segment_duration))
                step = len(windows) // (n_segments + 1)
                change_points = [step * (i + 1) for i in range(n_segments)]
        
        # Remove the last point (end of series) and filter by minimum duration
        change_points = [cp for cp in change_points[:-1] if cp > 0]
        
        # Filter change points by minimum segment duration
        filtered_change_points = []
        last_point = 0
        
        for cp in change_points:
            # Check if segment meets minimum duration
            if cp < len(windows):
                segment_duration = windows[cp]["end_time"] - windows[last_point]["start_time"]
                if segment_duration >= self.min_segment_duration:
                    filtered_change_points.append(cp)
                    last_point = cp
        
        print(f"✅ Detected {len(filtered_change_points)} topic boundaries")
        self.change_points = filtered_change_points
        return filtered_change_points

    def create_topic_segments(self, windows: List[Dict[str, Any]], change_points: List[int]) -> List[Dict[str, Any]]:
        """
        Create final topic-based segments from change points.
        
        Args:
            windows: Window metadata
            change_points: Change point indices
            
        Returns:
            List of topic segments
        """
        print("📝 Creating topic segments...")
        
        segments = []
        boundaries = [0] + change_points + [len(windows)]
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            if start_idx >= len(windows) or end_idx > len(windows):
                continue
                
            # Combine text from all windows in this segment
            segment_windows = windows[start_idx:end_idx]
            segment_text = " ".join([w["text"] for w in segment_windows])
            
            segment = {
                "segment_id": i + 1,
                "start_time": segment_windows[0]["start_time"],
                "end_time": segment_windows[-1]["end_time"],
                "duration": segment_windows[-1]["end_time"] - segment_windows[0]["start_time"],
                "text": segment_text,
                "window_count": len(segment_windows),
                "windows": segment_windows,
                "timestamp_readable": self._format_timestamp(
                    segment_windows[0]["start_time"],
                    segment_windows[-1]["end_time"]
                )
            }
            segments.append(segment)
        
        # Post-process segments to enforce duration constraints
        segments = self._enforce_duration_constraints(segments)
        
        print(f"✅ Created {len(segments)} topic segments")
        self.segments = segments
        return segments

    def _enforce_duration_constraints(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-process segments to enforce minimum and maximum duration constraints.
        
        Args:
            segments: Initial topic segments
            
        Returns:
            Adjusted segments meeting duration constraints
        """
        print("🔧 Enforcing duration constraints...")
        
        if not segments:
            return segments
        
        adjusted_segments = []
        i = 0
        
        while i < len(segments):
            current_segment = segments[i].copy()
            
            # Check if segment is too short
            if current_segment["duration"] < self.min_segment_duration:
                # Try to merge with next segment
                if i + 1 < len(segments):
                    next_segment = segments[i + 1]
                    combined_duration = next_segment["end_time"] - current_segment["start_time"]
                    
                    # Merge if combined duration doesn't exceed maximum
                    if combined_duration <= self.max_segment_duration:
                        merged_segment = {
                            "segment_id": current_segment["segment_id"],
                            "start_time": current_segment["start_time"],
                            "end_time": next_segment["end_time"],
                            "duration": combined_duration,
                            "text": current_segment["text"] + " " + next_segment["text"],
                            "window_count": current_segment["window_count"] + next_segment["window_count"],
                            "windows": current_segment["windows"] + next_segment["windows"],
                            "timestamp_readable": self._format_timestamp(
                                current_segment["start_time"],
                                next_segment["end_time"]
                            )
                        }
                        adjusted_segments.append(merged_segment)
                        i += 2  # Skip next segment as it's been merged
                        continue
                    else:
                        # If merging would exceed max, extend current segment to minimum
                        current_segment["end_time"] = current_segment["start_time"] + self.min_segment_duration
                        current_segment["duration"] = self.min_segment_duration
                        current_segment["timestamp_readable"] = self._format_timestamp(
                            current_segment["start_time"],
                            current_segment["end_time"]
                        )
                else:
                    # Last segment - extend to minimum duration
                    current_segment["end_time"] = current_segment["start_time"] + self.min_segment_duration
                    current_segment["duration"] = self.min_segment_duration
                    current_segment["timestamp_readable"] = self._format_timestamp(
                        current_segment["start_time"],
                        current_segment["end_time"]
                    )
            
            # Check if segment is too long
            elif current_segment["duration"] > self.max_segment_duration:
                # Split segment into multiple parts
                total_duration = current_segment["duration"]
                n_parts = int(np.ceil(total_duration / self.max_segment_duration))
                part_duration = total_duration / n_parts
                
                for part_idx in range(n_parts):
                    part_start = current_segment["start_time"] + (part_idx * part_duration)
                    part_end = min(part_start + part_duration, current_segment["end_time"])
                    
                    # Estimate text for this part (simple split)
                    text_start = int((part_idx / n_parts) * len(current_segment["text"]))
                    text_end = int(((part_idx + 1) / n_parts) * len(current_segment["text"]))
                    part_text = current_segment["text"][text_start:text_end]
                    
                    part_segment = {
                        "segment_id": f"{current_segment['segment_id']}.{part_idx + 1}",
                        "start_time": part_start,
                        "end_time": part_end,
                        "duration": part_end - part_start,
                        "text": part_text,
                        "window_count": current_segment["window_count"] // n_parts,
                        "windows": current_segment["windows"][
                            (part_idx * len(current_segment["windows"]) // n_parts):
                            ((part_idx + 1) * len(current_segment["windows"]) // n_parts)
                        ],
                        "timestamp_readable": self._format_timestamp(part_start, part_end)
                    }
                    adjusted_segments.append(part_segment)
                
                i += 1
                continue
            
            # Segment duration is within bounds
            adjusted_segments.append(current_segment)
            i += 1
        
        # Renumber segments
        for idx, segment in enumerate(adjusted_segments):
            segment["segment_id"] = idx + 1
        
        original_count = len(segments)
        final_count = len(adjusted_segments)
        
        if final_count != original_count:
            print(f"🔄 Adjusted segments: {original_count} → {final_count}")
        
        # Verify all segments meet constraints
        violations = 0
        for segment in adjusted_segments:
            if segment["duration"] < self.min_segment_duration or segment["duration"] > self.max_segment_duration:
                violations += 1
        
        if violations > 0:
            print(f"⚠️  {violations} segments still violate duration constraints")
        else:
            print(f"✅ All segments meet duration constraints ({self.min_segment_duration}s - {self.max_segment_duration}s)")
        
        return adjusted_segments

    def apply_topic_modeling(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply BERTopic to identify and label topics.
        
        Args:
            segments: Topic segments
            
        Returns:
            Topic modeling results
        """
        print("🏷️ Applying topic modeling...")
        
        texts = [segment["text"] for segment in segments]
        
        # Fit BERTopic
        topics, probs = self.topic_model.fit_transform(texts)
        
        # Get topic information
        topic_info = self.topic_model.get_topic_info()
        
        # Add topic information to segments
        for i, segment in enumerate(segments):
            topic_id = topics[i]
            segment["topic_id"] = topic_id
            segment["topic_probability"] = probs[i] if probs is not None else None
            
            # Get topic label
            if topic_id != -1:  # -1 is outlier topic
                topic_words = self.topic_model.get_topic(topic_id)
                topic_label = ", ".join([word for word, _ in topic_words[:3]])
                segment["topic_label"] = topic_label
            else:
                segment["topic_label"] = "Outlier/Mixed"
        
        results = {
            "segments": segments,
            "topic_info": topic_info,
            "topics": topics,
            "probabilities": probs
        }
        
        print(f"✅ Identified {len(topic_info)} unique topics")
        self.topics = results
        return results

    def visualize_topic_evolution(self, segments: List[Dict[str, Any]], output_dir: Path):
        """
        Create visualizations of topic evolution over time.
        
        Args:
            segments: Topic segments with topic information
            output_dir: Directory to save visualizations
        """
        print("📊 Creating topic evolution visualizations...")
        
        output_dir.mkdir(exist_ok=True)
        
        # 1. Topic timeline
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        topic_ids = sorted(list(set([s["topic_id"] for s in segments])))
        
        for i, segment in enumerate(segments):
            topic_id = segment["topic_id"]
            color_idx = topic_ids.index(topic_id) % len(colors)
            
            fig.add_trace(go.Scatter(
                x=[segment["start_time"], segment["end_time"]],
                y=[topic_id, topic_id],
                mode='lines+markers',
                name=f'Topic {topic_id}: {segment["topic_label"]}',
                line=dict(color=colors[color_idx], width=10),
                hovertext=f"Segment {segment['segment_id']}<br>"
                         f"Duration: {segment['duration']:.1f}s<br>"
                         f"Topic: {segment['topic_label']}<br>"
                         f"Text: {segment['text'][:100]}...",
                showlegend=i == 0 or segment["topic_id"] != segments[i-1]["topic_id"]
            ))
        
        fig.update_layout(
            title="Topic Evolution Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Topic ID",
            height=600,
            hovermode='closest'
        )
        
        fig.write_html(output_dir / "topic_timeline.html")
        
        # 2. Topic similarity heatmap
        if len(segments) > 1:
            segment_embeddings = self.embedding_model.encode([s["text"] for s in segments])
            similarity_matrix = cosine_similarity(segment_embeddings)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                similarity_matrix,
                annot=False,
                cmap='viridis',
                ax=ax,
                xticklabels=[f"S{s['segment_id']}" for s in segments],
                yticklabels=[f"S{s['segment_id']}" for s in segments]
            )
            ax.set_title("Segment Similarity Matrix")
            plt.tight_layout()
            plt.savefig(output_dir / "segment_similarity.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Topic word clouds (if topics were generated)
        if hasattr(self.topic_model, 'get_topic_info'):
            topic_info = self.topic_model.get_topic_info()
            if len(topic_info) > 1:  # More than just outlier topic
                fig = self.topic_model.visualize_barchart(top_n_topics=min(8, len(topic_info)-1))
                fig.write_html(output_dir / "topic_words.html")
        
        print(f"✅ Visualizations saved to {output_dir}")

    def analyze_topic_coherence(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze topic coherence and quality metrics.
        
        Args:
            segments: Topic segments
            
        Returns:
            Coherence analysis results
        """
        print("🎯 Analyzing topic coherence...")
        
        analysis = {
            "total_segments": len(segments),
            "average_duration": np.mean([s["duration"] for s in segments]),
            "duration_std": np.std([s["duration"] for s in segments]),
            "topic_distribution": {},
            "duration_by_topic": {}
        }
        
        # Topic distribution
        topic_counts = {}
        topic_durations = {}
        
        for segment in segments:
            topic_id = segment["topic_id"]
            duration = segment["duration"]
            
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
            if topic_id not in topic_durations:
                topic_durations[topic_id] = []
            topic_durations[topic_id].append(duration)
        
        analysis["topic_distribution"] = topic_counts
        analysis["duration_by_topic"] = {
            topic: {
                "mean": np.mean(durations),
                "std": np.std(durations),
                "count": len(durations)
            }
            for topic, durations in topic_durations.items()
        }
        
        # Calculate topic transition frequency
        transitions = {}
        for i in range(1, len(segments)):
            prev_topic = segments[i-1]["topic_id"]
            curr_topic = segments[i]["topic_id"]
            
            if prev_topic != curr_topic:
                transition = f"{prev_topic} -> {curr_topic}"
                transitions[transition] = transitions.get(transition, 0) + 1
        
        analysis["topic_transitions"] = transitions
        
        print(f"✅ Analysis complete: {len(topic_counts)} unique topics")
        return analysis

    def _format_timestamp(self, start_time: float, end_time: float) -> str:
        """Format timestamp for readability."""
        start_min, start_sec = divmod(int(start_time), 60)
        end_min, end_sec = divmod(int(end_time), 60)
        return f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        import re
        # Replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Replace full-width characters
        sanitized = sanitized.replace('：', '_')
        sanitized = sanitized.replace('？', '_')
        sanitized = sanitized.replace('！', '_')
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Limit length to avoid filesystem limits
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized

    def process_video_transcript(self, transcript_path: str, output_dir: str = "./topic_analysis") -> Dict[str, Any]:
        """
        Complete pipeline for topic-based video segmentation.
        
        Args:
            transcript_path: Path to transcript JSON file
            output_dir: Directory to save results
            
        Returns:
            Complete analysis results
        """
        print("🚀 Starting advanced topic segmentation pipeline...")
        
        # Create sanitized output directory
        transcript_file = Path(transcript_path)
        sanitized_name = self._sanitize_filename(transcript_file.stem)
        output_path = Path(output_dir) / sanitized_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        print(f"📄 Loaded transcript: {transcript_data['video_name']}")
        print(f"💾 Output directory: {output_path}")
        
        # Step 1: Create sliding windows
        windows = self.create_sliding_windows(transcript_data)
        
        # Step 2: Generate embeddings
        embeddings = self.generate_embeddings(windows)
        
        # Step 3: Detect topic boundaries
        change_points = self.detect_topic_boundaries(embeddings, windows)
        
        # Step 4: Create topic segments
        segments = self.create_topic_segments(windows, change_points)
        
        # Step 5: Apply topic modeling
        topic_results = self.apply_topic_modeling(segments)
        
        # Step 6: Analyze coherence
        coherence_analysis = self.analyze_topic_coherence(segments)
        
        # Step 7: Create visualizations
        self.visualize_topic_evolution(segments, output_path)
        
        # Save results
        results = {
            "video_info": {
                "name": transcript_data["video_name"],
                "duration": transcript_data.get("duration", 0),
                "processed_at": datetime.now().isoformat()
            },
            "windows": windows,
            "change_points": change_points,
            "segments": segments,
            "topic_results": topic_results,
            "coherence_analysis": coherence_analysis,
            "processing_params": {
                "window_size": self.window_size,
                "overlap_ratio": self.overlap_ratio,
                "min_segment_duration": self.min_segment_duration,
                "max_segment_duration": self.max_segment_duration,
                "embedding_model": self.embedding_model_name
            }
        }
        
        # Save to file
        results_file = output_path / "topic_analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = json.loads(json.dumps(results, default=str))
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Analysis complete! Results saved to {results_file}")
        return results

def main():
    """Example usage of the advanced topic segmentation system."""
    # Example usage
    segmenter = AdvancedTopicSegmenter(
        embedding_model="all-MiniLM-L6-v2",
        window_size=30,  # 30-second windows
        overlap_ratio=0.5,  # 50% overlap
        min_segment_duration=60,  # 1 minute minimum
        max_segment_duration=180  # 3 minutes maximum
    )
    
    # Process transcript files
    transcript_dir = Path("./processed/transcripts")
    
    if transcript_dir.exists():
        transcript_files = list(transcript_dir.glob("*_transcript.json"))
        
        if transcript_files:
            print(f"Found {len(transcript_files)} transcript files")
            
            for transcript_file in transcript_files[:2]:  # Process first 2 files as example
                try:
                    results = segmenter.process_video_transcript(
                        str(transcript_file),
                        "./topic_analysis"
                    )
                    
                    print(f"\n📊 Results for {transcript_file.name}:")
                    print(f"   Segments: {len(results['segments'])}")
                    print(f"   Topics: {len(results['topic_results']['topic_info'])}")
                    print(f"   Avg duration: {results['coherence_analysis']['average_duration']:.1f}s")
                    
                except Exception as e:
                    print(f"❌ Error processing {transcript_file.name}: {e}")
        else:
            print("❌ No transcript files found. Please run process_video.py first.")
    else:
        print("❌ Transcript directory not found. Please run process_video.py first.")

if __name__ == "__main__":
    main() 