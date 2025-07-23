#!/usr/bin/env python3
"""
Demonstration: Topic-Based vs Time-Based Video Segmentation

This script demonstrates the difference between basic time-based segmentation
and advanced topic-based segmentation using embeddings and change point detection.

Usage:
    python demo_topic_segmentation.py
"""

import json
from pathlib import Path
from advanced_segmentation import AdvancedTopicSegmenter

# For visualization - try multiple options in case some aren't available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available - will create text-based visualization")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def compare_segmentation_approaches():
    """Compare different segmentation approaches on the same video."""
    
    print("🔬 Comparing Segmentation Approaches")
    print("=" * 50)
    
    # Find a sample transcript
    transcript_dir = Path("./processed/transcripts")
    transcript_files = list(transcript_dir.glob("*_transcript.json"))
    
    if not transcript_files:
        print("❌ No transcript files found. Please run process_video.py first.")
        return
    
    sample_file = transcript_files[0]
    print(f"📄 Using sample: {sample_file.name}")
    
    # Load transcript
    with open(sample_file, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    print(f"📊 Video duration: {transcript_data.get('duration', 0)/60:.1f} minutes")
    print(f"📝 Original segments: {len(transcript_data['segments'])}")
    
    # 1. Time-based segmentation (current approach)
    print("\n1️⃣ Time-based Segmentation (Current):")
    time_segments = create_time_based_segments(transcript_data, target_duration=90)
    print(f"   Created {len(time_segments)} segments")
    print(f"   Avg duration: {sum(s['duration'] for s in time_segments)/len(time_segments):.1f}s")
    
    # 2. Advanced topic-based segmentation
    print("\n2️⃣ Topic-based Segmentation (Advanced):")
    segmenter = AdvancedTopicSegmenter(
        window_size=30,
        overlap_ratio=0.5,
        min_segment_duration=60,
        max_segment_duration=180
    )
    
    # Process with advanced method
    try:
        results = segmenter.process_video_transcript(
            str(sample_file),
            "./demo_comparison"
        )
    except Exception as e:
        print(f"❌ Error in advanced segmentation: {e}")
        return
    
    topic_segments = results['segments']
    print(f"   Created {len(topic_segments)} segments")
    print(f"   Avg duration: {sum(s['duration'] for s in topic_segments)/len(topic_segments):.1f}s")
    print(f"   Unique topics: {len(set(s['topic_id'] for s in topic_segments))}")
    
    # 3. Create detailed segment breakdown
    create_segment_breakdown(time_segments, topic_segments, transcript_data)
    
    # 4. Visualize comparison
    visualize_comparison(time_segments, topic_segments, transcript_data)
    
    # 5. Show topic examples
    show_topic_examples(topic_segments)

def create_time_based_segments(transcript_data, target_duration=90):
    """Create time-based segments for comparison."""
    segments = transcript_data["segments"]
    if not segments:
        return []
    
    time_segments = []
    current_segment = {
        "start_time": segments[0]["start"],
        "end_time": segments[0]["end"],
        "text": segments[0]["text"],
        "source_segments": [segments[0]["id"]]
    }
    
    for i, segment in enumerate(segments[1:], 1):
        current_duration = current_segment["end_time"] - current_segment["start_time"]
        
        if current_duration >= target_duration:
            # Finalize current segment
            current_segment["duration"] = current_segment["end_time"] - current_segment["start_time"]
            time_segments.append(current_segment)
            
            # Start new segment
            current_segment = {
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"],
                "source_segments": [segment["id"]]
            }
        else:
            # Extend current segment
            current_segment["end_time"] = segment["end"]
            current_segment["text"] += " " + segment["text"]
            current_segment["source_segments"].append(segment["id"])
    
    # Add the last segment
    if current_segment["text"]:
        current_segment["duration"] = current_segment["end_time"] - current_segment["start_time"]
        time_segments.append(current_segment)
    
    return time_segments

def create_segment_breakdown(time_segments, topic_segments, transcript_data):
    """Create detailed breakdown showing start times and durations for each segment."""
    
    print("\n📋 Detailed Segment Breakdown")
    print("=" * 80)
    
    # Show side-by-side comparison of first 10 segments
    print(f"{'TIME-BASED SEGMENTS':<40} {'TOPIC-BASED SEGMENTS':<40}")
    print(f"{'-'*40} {'-'*40}")
    print(f"{'ID':<3} {'Start':<8} {'Duration':<8} {'Text...':<20} {'ID':<3} {'Start':<8} {'Duration':<8} {'Topic':<20}")
    print(f"{'-'*80}")
    
    max_segments = max(len(time_segments), len(topic_segments))
    
    for i in range(min(15, max_segments)):  # Show first 15 segments
        # Time-based segment info
        if i < len(time_segments):
            time_seg = time_segments[i]
            time_info = f"{i+1:<3} {time_seg['start_time']:<8.1f} {time_seg['duration']:<8.1f} {time_seg['text'][:20]:<20}"
        else:
            time_info = f"{'':^40}"
        
        # Topic-based segment info  
        if i < len(topic_segments):
            topic_seg = topic_segments[i]
            topic_label = topic_seg.get('topic_label', f"Topic {topic_seg.get('topic_id', 'N/A')}")
            topic_info = f"{i+1:<3} {topic_seg['start_time']:<8.1f} {topic_seg['duration']:<8.1f} {topic_label[:20]:<20}"
        else:
            topic_info = f"{'':^40}"
        
        print(f"{time_info} {topic_info}")
    
    if max_segments > 15:
        print(f"... and {max_segments - 15} more segments")
    
    # Show timing analysis
    print(f"\n📊 Timing Analysis")
    print(f"{'='*80}")
    
    # Calculate overlap between approaches
    overlaps = []
    alignment_score = 0
    
    for time_seg in time_segments:
        best_overlap = 0
        for topic_seg in topic_segments:
            # Calculate overlap between segments
            overlap_start = max(time_seg['start_time'], topic_seg['start_time'])
            overlap_end = min(
                time_seg['start_time'] + time_seg['duration'],
                topic_seg['start_time'] + topic_seg['duration']
            )
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
        
        overlaps.append(best_overlap)
        alignment_score += best_overlap / time_seg['duration']
    
    avg_alignment = alignment_score / len(time_segments) if time_segments else 0
    
    print(f"🔗 Segment Alignment:")
    print(f"   • Average alignment score: {avg_alignment:.2f} (1.0 = perfect alignment)")
    print(f"   • Average overlap: {sum(overlaps)/len(overlaps):.1f}s per segment")
    
    # Show boundary differences
    boundary_diffs = []
    for i, time_seg in enumerate(time_segments):
        # Find closest topic segment boundary
        time_start = time_seg['start_time']
        closest_topic_start = min(
            (abs(time_start - topic_seg['start_time']), topic_seg['start_time']) 
            for topic_seg in topic_segments
        )[1]
        
        boundary_diffs.append(abs(time_start - closest_topic_start))
    
    print(f"🎯 Boundary Analysis:")
    print(f"   • Average boundary difference: {sum(boundary_diffs)/len(boundary_diffs):.1f}s")
    print(f"   • Max boundary difference: {max(boundary_diffs):.1f}s")
    print(f"   • Boundaries within 30s: {sum(1 for d in boundary_diffs if d <= 30)} of {len(boundary_diffs)}")

def visualize_comparison(time_segments, topic_segments, transcript_data):
    """Create comprehensive A/B timeline visualization comparing the two approaches."""
    
    print("\n📊 Creating A/B Timeline Visualization")
    print("=" * 50)
    
    # Always create text-based visualization first
    create_text_timeline(time_segments, topic_segments, transcript_data)
    
    # Try interactive visualization with Plotly (preferred)
    if PLOTLY_AVAILABLE:
        create_interactive_timeline(time_segments, topic_segments, transcript_data)
    
    # Fallback to matplotlib if available
    elif MATPLOTLIB_AVAILABLE:
        create_matplotlib_timeline(time_segments, topic_segments, transcript_data)
    
    else:
        print("💡 For better visualizations, install: pip install plotly matplotlib")

def create_text_timeline(time_segments, topic_segments, transcript_data):
    """Create a detailed text-based timeline comparison."""
    
    print("\n📋 Text-Based Timeline Comparison")
    print("=" * 60)
    
    total_duration = transcript_data.get('duration', 0)
    if total_duration <= 0 and topic_segments:
        total_duration = max(s['start_time'] + s['duration'] for s in topic_segments)
    
    print(f"🎬 Video: {transcript_data.get('video_name', 'Unknown')}")
    print(f"⏱️  Total Duration: {total_duration/60:.1f} minutes ({total_duration:.0f} seconds)")
    print(f"📊 Time-based: {len(time_segments)} segments | Topic-based: {len(topic_segments)} segments")
    
    # Create timeline comparison
    print(f"\n{'='*80}")
    print(f"{'TIMELINE COMPARISON':<80}")
    print(f"{'='*80}")
    print(f"{'Time (min:sec)':<12} {'Time-Based Segments':<35} {'Topic-Based Segments':<33}")
    print(f"{'-'*80}")
    
    # Create time markers every 30 seconds
    time_markers = list(range(0, int(total_duration) + 30, 30))
    
    for time_point in time_markers[:20]:  # Show first 10 minutes
        time_str = f"{time_point//60:02d}:{time_point%60:02d}"
        
        # Find active time-based segment
        time_segment_info = "·"
        for i, seg in enumerate(time_segments):
            if seg['start_time'] <= time_point < seg['start_time'] + seg['duration']:
                time_segment_info = f"Segment {i+1} ({seg['duration']:.0f}s)"
                break
        
        # Find active topic-based segment
        topic_segment_info = "·"
        for i, seg in enumerate(topic_segments):
            if seg['start_time'] <= time_point < seg['start_time'] + seg['duration']:
                topic_label = seg.get('topic_label', f"Topic {seg.get('topic_id', 'Unknown')}")
                topic_segment_info = f"S{i+1}: {topic_label[:20]}..."
                break
        
        print(f"{time_str:<12} {time_segment_info:<35} {topic_segment_info:<33}")
    
    # Show segment statistics
    print(f"\n{'='*80}")
    print(f"{'SEGMENT STATISTICS':<80}")
    print(f"{'='*80}")
    
    # Time-based stats
    time_durations = [s['duration'] for s in time_segments]
    print(f"📊 Time-Based Segmentation:")
    print(f"   • Count: {len(time_segments)} segments")
    print(f"   • Average duration: {sum(time_durations)/len(time_durations):.1f}s ({sum(time_durations)/len(time_durations)/60:.1f}m)")
    print(f"   • Duration range: {min(time_durations):.1f}s - {max(time_durations):.1f}s")
    print(f"   • Total coverage: {sum(time_durations):.1f}s ({sum(time_durations)/60:.1f}m)")
    
    # Topic-based stats
    topic_durations = [s['duration'] for s in topic_segments]
    topic_ids = set(s.get('topic_id', -1) for s in topic_segments)
    print(f"\n🎯 Topic-Based Segmentation:")
    print(f"   • Count: {len(topic_segments)} segments")
    print(f"   • Unique topics: {len(topic_ids)}")
    print(f"   • Average duration: {sum(topic_durations)/len(topic_durations):.1f}s ({sum(topic_durations)/len(topic_durations)/60:.1f}m)")
    print(f"   • Duration range: {min(topic_durations):.1f}s - {max(topic_durations):.1f}s")
    print(f"   • Total coverage: {sum(topic_durations):.1f}s ({sum(topic_durations)/60:.1f}m)")

def create_interactive_timeline(time_segments, topic_segments, transcript_data):
    """Create interactive timeline with Plotly."""
    
    print("\n🎨 Creating Interactive Timeline (Plotly)")
    
    # Create subplot with shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Time-Based Segmentation', 'Topic-Based Segmentation'),
        row_heights=[0.5, 0.5]
    )
    
    # Colors for segments
    colors = px.colors.qualitative.Set3
    
    # Add time-based segments
    for i, segment in enumerate(time_segments):
        fig.add_trace(
            go.Scatter(
                x=[segment['start_time'], segment['start_time'] + segment['duration']],
                y=[1, 1],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=20),
                name=f'Time Seg {i+1}',
                hovertext=f"Segment {i+1}<br>Start: {segment['start_time']:.1f}s<br>Duration: {segment['duration']:.1f}s<br>Text: {segment['text'][:100]}...",
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Add topic-based segments
    topic_colors = {}
    color_idx = 0
    
    for i, segment in enumerate(topic_segments):
        topic_id = segment.get('topic_id', -1)
        if topic_id not in topic_colors:
            topic_colors[topic_id] = colors[color_idx % len(colors)]
            color_idx += 1
        
        topic_label = segment.get('topic_label', f'Topic {topic_id}')
        
        fig.add_trace(
            go.Scatter(
                x=[segment['start_time'], segment['start_time'] + segment['duration']],
                y=[1, 1],
                mode='lines',
                line=dict(color=topic_colors[topic_id], width=20),
                name=f'Topic {topic_id}',
                hovertext=f"Segment {i+1}<br>Topic: {topic_label}<br>Start: {segment['start_time']:.1f}s<br>Duration: {segment['duration']:.1f}s<br>Text: {segment['text'][:100]}...",
                showlegend=i == 0 or topic_id not in [s.get('topic_id', -1) for s in topic_segments[:i]]
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"A/B Timeline Comparison: {transcript_data.get('video_name', 'Video Segmentation')}",
        xaxis_title="Time (seconds)",
        height=600,
        hovermode='closest'
    )
    
    # Update y-axes to show segment labels
    fig.update_yaxes(title_text="Segments", row=1, col=1, range=[0.5, 1.5], showticklabels=False)
    fig.update_yaxes(title_text="Topics", row=2, col=1, range=[0.5, 1.5], showticklabels=False)
    
    # Save interactive plot
    output_file = Path("./demo_comparison/interactive_timeline.html")
    fig.write_html(output_file)
    print(f"✅ Interactive timeline saved: {output_file}")
    
    # Also create a static comparison chart
    create_segment_comparison_chart(time_segments, topic_segments, transcript_data)

def create_segment_comparison_chart(time_segments, topic_segments, transcript_data):
    """Create a comparison chart showing segment distributions."""
    
    # Create duration distribution comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Time-Based Duration Distribution', 'Topic-Based Duration Distribution',
                       'Segment Count Over Time', 'Topic Distribution'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    # Duration histograms
    time_durations = [s['duration'] for s in time_segments]
    topic_durations = [s['duration'] for s in topic_segments]
    
    fig.add_trace(
        go.Histogram(x=time_durations, name='Time-Based', nbinsx=10),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=topic_durations, name='Topic-Based', nbinsx=10),
        row=1, col=2
    )
    
    # Segment count over time
    time_points = []
    time_counts = []
    topic_counts = []
    
    total_duration = transcript_data.get('duration', max(s['start_time'] + s['duration'] for s in topic_segments))
    
    for t in range(0, int(total_duration), 60):  # Every minute
        time_active = sum(1 for s in time_segments if s['start_time'] <= t < s['start_time'] + s['duration'])
        topic_active = sum(1 for s in topic_segments if s['start_time'] <= t < s['start_time'] + s['duration'])
        
        time_points.append(t)
        time_counts.append(time_active)
        topic_counts.append(topic_active)
    
    fig.add_trace(
        go.Scatter(x=time_points, y=time_counts, name='Time-Based', mode='lines+markers'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=time_points, y=topic_counts, name='Topic-Based', mode='lines+markers'),
        row=2, col=1
    )
    
    # Topic distribution pie chart
    topic_counts = {}
    for segment in topic_segments:
        topic_id = segment.get('topic_id', -1)
        topic_label = segment.get('topic_label', f'Topic {topic_id}')
        topic_counts[topic_label] = topic_counts.get(topic_label, 0) + 1
    
    fig.add_trace(
        go.Pie(labels=list(topic_counts.keys()), values=list(topic_counts.values()), name="Topics"),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Detailed Segmentation Analysis",
        height=800,
        showlegend=True
    )
    
    # Save analysis chart
    output_file = Path("./demo_comparison/segmentation_analysis.html")
    fig.write_html(output_file)
    print(f"✅ Analysis chart saved: {output_file}")

def create_matplotlib_timeline(time_segments, topic_segments, transcript_data):
    """Create timeline visualization with matplotlib (fallback)."""
    
    print("\n📊 Creating Timeline with Matplotlib")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    
    # Time-based segments timeline
    for i, segment in enumerate(time_segments):
        ax1.barh(0, segment["duration"], left=segment["start_time"], 
                height=0.4, alpha=0.8, color=f'C{i%10}', 
                edgecolor='white', linewidth=1)
        
        # Add segment number
        mid_point = segment["start_time"] + segment["duration"] / 2
        ax1.text(mid_point, 0, f'{i+1}', ha='center', va='center', fontweight='bold', fontsize=8)
    
    ax1.set_title("A: Time-Based Segmentation", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Segments")
    ax1.set_ylim(-0.5, 0.5)
    ax1.grid(True, alpha=0.3)
    
    # Topic-based segments timeline
    topic_colors = {}
    color_idx = 0
    
    for i, segment in enumerate(topic_segments):
        topic_id = segment.get("topic_id", -1)
        if topic_id not in topic_colors:
            topic_colors[topic_id] = f'C{color_idx%10}'
            color_idx += 1
        
        color = topic_colors[topic_id]
        
        ax2.barh(0, segment["duration"], left=segment["start_time"], 
                height=0.4, alpha=0.8, color=color,
                edgecolor='white', linewidth=1)
        
        # Add segment number
        mid_point = segment["start_time"] + segment["duration"] / 2
        ax2.text(mid_point, 0, f'{i+1}', ha='center', va='center', fontweight='bold', fontsize=8)
    
    ax2.set_title("B: Topic-Based Segmentation", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Topics")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add time markers
    total_duration = transcript_data.get('duration', max(s['start_time'] + s['duration'] for s in topic_segments))
    time_ticks = list(range(0, int(total_duration), 300))  # Every 5 minutes
    for ax in [ax1, ax2]:
        ax.set_xticks(time_ticks)
        ax.set_xticklabels([f"{t//60:02d}:{t%60:02d}" for t in time_ticks])
    
    plt.tight_layout()
    plt.savefig("./demo_comparison/timeline_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Timeline saved: ./demo_comparison/timeline_comparison.png")
    
    # Show if running interactively
    try:
        plt.show()
    except:
        pass  # Skip if can't display

def show_topic_examples(topic_segments):
    """Show examples of identified topics."""
    
    print("\n🏷️ Topic Examples:")
    print("=" * 50)
    
    # Group by topic
    topics = {}
    for segment in topic_segments:
        topic_id = segment.get("topic_id", -1)
        topic_label = segment.get("topic_label", f"Topic {topic_id}")
        
        if topic_id not in topics:
            topics[topic_id] = {
                "label": topic_label,
                "segments": [],
                "total_duration": 0
            }
        
        topics[topic_id]["segments"].append(segment)
        topics[topic_id]["total_duration"] += segment["duration"]
    
    # Show top topics by duration
    sorted_topics = sorted(topics.items(), key=lambda x: x[1]["total_duration"], reverse=True)
    
    for i, (topic_id, topic_info) in enumerate(sorted_topics[:5]):
        print(f"\n🎯 Topic {topic_id}: {topic_info['label']}")
        print(f"   Duration: {topic_info['total_duration']:.1f}s ({len(topic_info['segments'])} segments)")
        
        # Show a sample segment
        sample_segment = topic_info['segments'][0]
        sample_text = sample_segment['text'][:200] + "..." if len(sample_segment['text']) > 200 else sample_segment['text']
        print(f"   Sample: {sample_text}")
        print(f"   Time: {sample_segment.get('timestamp_readable', 'N/A')}")

def analyze_topic_quality():
    """Analyze the quality of topic segmentation."""
    
    print("\n📊 Topic Quality Analysis")
    print("=" * 30)
    
    # This would include metrics like:
    # - Topic coherence scores
    # - Segment duration consistency
    # - Topic transition smoothness
    # - Semantic similarity within topics
    
    print("📈 Metrics to consider:")
    print("   • Intra-topic similarity (how similar segments within same topic)")
    print("   • Inter-topic diversity (how different topics are from each other)")
    print("   • Temporal coherence (smooth topic transitions)")
    print("   • Duration consistency (segments have reasonable lengths)")
    print("   • Boundary precision (topic changes align with content changes)")

def main():
    """Run the comprehensive A/B timeline comparison demonstration."""
    
    print("🎬 A/B Timeline Comparison: Time-Based vs Topic-Based Segmentation")
    print("=" * 70)
    print("This demo shows side-by-side timeline visualization of segment distribution")
    print("showing exactly where each segment starts and how long it lasts.")
    print()
    
    # Create output directory
    output_dir = Path("./demo_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Run comprehensive comparison
    try:
        compare_segmentation_approaches()
        
        # Analyze quality
        analyze_topic_quality()
        
        print("\n✅ A/B Timeline Comparison Complete!")
        print("=" * 50)
        print(f"📁 Results saved to: {output_dir.absolute()}")
        
        # List generated files
        generated_files = list(output_dir.glob("*"))
        if generated_files:
            print("\n📊 Generated visualizations:")
            for file in generated_files:
                if file.suffix == '.html':
                    print(f"   🌐 {file.name} (open in browser)")
                elif file.suffix == '.png':
                    print(f"   🖼️  {file.name} (image)")
                else:
                    print(f"   📄 {file.name}")
        
        print("\n💡 Key Timeline Insights:")
        print("   ✅ Time-based: Fixed duration segments (predictable but arbitrary cuts)")
        print("   ✅ Topic-based: Adaptive segments (natural topic boundaries)")
        print("   ✅ Visual comparison shows how segments align with content flow")
        print("   ✅ Interactive timeline allows detailed segment exploration")
        
        if PLOTLY_AVAILABLE:
            print("\n🎯 Next Steps:")
            print(f"   1. Open interactive_timeline.html in your browser")
            print(f"   2. Hover over segments to see detailed information")
            print(f"   3. Compare how boundaries align with actual content")
        elif MATPLOTLIB_AVAILABLE:
            print("\n🎯 Next Steps:")
            print(f"   1. View timeline_comparison.png for visual comparison")
            print(f"   2. Install plotly for interactive visualization: pip install plotly")
        else:
            print("\n🎯 For Better Visualizations:")
            print(f"   pip install plotly matplotlib")
            
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("💡 Make sure you have transcript files in ./processed/transcripts/")
        print("💡 Run process_video.py first to generate transcripts")

if __name__ == "__main__":
    main() 