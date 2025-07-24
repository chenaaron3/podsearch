# Download all videos under a podcast

python3 scripts/video_downloader.py

# Process the video and embed the data

python3 scripts/process_video.py

# Search the video

python3 scripts/search_segments.py

# Update pip requirements

pip freeze -q -r ./scripts/requirements.txt | sed '/freeze/,$ d' > requirements-froze.txt
