from get_videos import get_videos, print_videos

def main():
    """Main function."""
    # Get videos from channel
    videos = get_videos("https://www.youtube.com/@TheDiaryOfACEO")
    # Download the videos so we can process them locally
    print_videos(videos)

if __name__ == "__main__":
    main()
