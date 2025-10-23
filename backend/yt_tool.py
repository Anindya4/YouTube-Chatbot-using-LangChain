import re
from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from backend.helper_fun import translate_large_text_parallel


@tool
def get_video_id(url: str) -> str:
    """
    This function retrieve the video id from a given youtube video url.
    Args:
        url: str - The url to the video.
    
    Returns:
        str - Returns the extracted id.
    """
    pattern = (
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    )
    match = re.search(pattern,url)
    if match:
        return match.group(1)
    return "No YouTube video ID found for this URL! Please check the URL"


@tool
def get_transcript(video_id : str) -> str:
    """
        Fetches the transcript for a YouTube video. It first tries to get the
        English transcript. If that fails, it tries to get a Hindi transcript
        and translates it to English.
        Args:
            video_id: str - The YouTube video ID.
        Returns:
            transcript: str - The full English transcript as a single text string, or an error message if unavailable.
    """
                        
    try:
        # First try to fetch english transcript:
        transcript_list = YouTubeTranscriptApi().fetch(video_id=video_id, languages=['en'])
        print("\nSuccessfuly fetched english transcript.\n")
        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        return "\n\nTranscripts are disabled for this video.\n\n"
    except Exception as e:
        print(f"\n\nError fetching english transcription: {e}. \n\n Trying Hindi...\n\n") 
        try: #Try to fetch hindi then translate to english
            transcript_list = YouTubeTranscriptApi().fetch(video_id=video_id, languages=['hi'])
            hindi_transcript = " ".join(chunk.text for chunk in transcript_list)
            print('\nSuccessfully get Hindi transcript. Now translating to english\n')
            english_transcript = translate_large_text_parallel(hindi_transcript)
            print('\nTranslation to is English complete\n')
            return english_transcript
        except Exception as e_inner:
            return f"Failed to fetch transcript in English or Hindi. Error: {e_inner}"
    
    

# def get_transcript(video_id: str) -> str:
#     """
#     Fetches the transcript for a YouTube video. It first tries 'en'.
#     If that fails, it tries 'hi', fetches its text, and translates it to 'en'.
#     """
#     try:
#         api = YouTubeTranscriptApi()
#         transcript_list = api.list_transcripts(video_id)

#         # --- (Try English block is unchanged) ---
#         try:
#             english_transcript = transcript_list.fetch(['en'])
#             print("Successfully found English transcript. Processing...")
#             transcript_data = english_transcript.fetch()
#             full_transcript = " ".join(chunk.text for chunk in transcript_data)
#             return full_transcript

#         except Exception as e:
#             print(f"Could not find English transcript: {e}. Trying Hindi fallback...")

#             # --- (Try Hindi block is updated) ---
#             hindi_transcript = transcript_list.fetch(['hi'])
#             print("Successfully found Hindi transcript. Fetching text...")
            
#             hindi_transcript_data = hindi_transcript.fetch()
#             hindi_text = " ".join(chunk.text for chunk in hindi_transcript_data)
            
#             # CHANGE 2: Explicitly pass source='hi' to the translator
#             print(f"Translating {len(hindi_text)} chars of Hindi text to English using MyMemory...")
#             english_transcript = translate_large_text_parallel(
#                 hindi_text, 
#                 target='en', 
#                 source='hi' # This new argument solves the language code problem
#             )
#             print("On-the-fly translation complete.")
#             return english_transcript

#     except TranscriptsDisabled:
#         return f"Transcripts are disabled for video ID: {video_id}."
#     except Exception as e:
#         return f"Failed to retrieve a suitable transcript for video ID {video_id}. It may not have an 'en' or 'hi' transcript available. Error: {e}"

            

