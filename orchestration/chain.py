from backend.yt_tool import get_video_id, get_transcript
from dotenv import load_dotenv

load_dotenv()

def get_transcript_from_url(url: str) -> str:
    """
    Directly fetches the transcript without using an agent.
    This is faster, more reliable, and returns the raw text.
    """
    print("Fetching video ID...")
    try:
        video_id = get_video_id.invoke(url)
        if "No YouTube video ID found" in video_id:
            return f"Error: {video_id}"
        
        print(f"Got video ID: {video_id}. Fetching transcript...")
        transcript_text = get_transcript.invoke(video_id)
        
        # This function now returns the RAW transcript, not a summary.
        return transcript_text
        
    except Exception as e:
        return f"An error occurred while fetching the transcript: {e}"

