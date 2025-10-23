# ALL THE HELPER FUNCTIONS WILL BE HERE:

from deep_translator import GoogleTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import textwrap
from langdetect import detect, LangDetectException


def translate_chunk(chunk:str, target: str = 'en') -> str:
    """Translate a single chunk with auto language detection.
    Args:
        chunk:str -> Text needed to be trasnlate.
        target:str -> Target language
    """
    try:
        source = detect(chunk)
        if not source:
            raise LangDetectException("Can't detect language code.")
        source = source.lower()
        if source == target:
            return chunk
        return GoogleTranslator(source=source, target=target).translate(chunk)
    except LangDetectException as e:
        print(f"\n\nError while detecting language automatically. Details\n\n: {e} \n\n\nReturning to auto detect via API...\n\n\n")
        return GoogleTranslator(source='auto', target=target).translate(chunk)
    except Exception as ex:
        print(f"\n\nError while translating language. Details\n\n: {ex}")
        return chunk
        
    


def translate_large_text_parallel(text, target='en', max_chars=1500, max_workers=5) -> str:
    """
    Translate large text by splitting into chunks and translating in parallel.
    
    :param text: str, input text of any length
    :param target: str, target language code (e.g., 'en', 'hi', 'de', 'es')
    :param max_chars: int, max characters per chunk (depends on backend)
    :param max_workers: int, number of threads to run in parallel
    :return: str, translated text
    """
    # Split text into manageable chunks
    chunks = textwrap.wrap(text, max_chars)
    translated_chunks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks to executor
        futures = [executor.submit(translate_chunk, chunk, target) for chunk in chunks]
        # Collect results as they finish
        for future in futures:
            translated_chunks.append(future.result())
    
    # Join translated chunks
    return " ".join(translated_chunks)


def split_transcript(transcript:str, chunk_size:int = 1500, chunk_overlap:int=200) -> List[Dict]:
    """
    Split transcript text into chunks for embedding/vectorstore.
    Args:
        transcripts: str -> The text the needed to be split.
        chunk_size : int -> The size of each spllited chunk.
        chunk_overlap:int -> The number of character to be common on each consecutive chunks.
    
    Returns:
        A list of dicts with 'text' and optional metadata.
    """
    spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = spliter.create_documents([transcript])
    return chunks


def format_doc(docs):
    """Clean the content from LLM for better readability"""
    clean_doc = '\n\n'.join(doc.page_content for doc in docs)
    return clean_doc