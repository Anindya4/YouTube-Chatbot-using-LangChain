import streamlit as st
from orchestration.chain import get_transcript_from_url
from backend.helper_fun import split_transcript, format_doc
from backend.vector_utils import vectorstore_from_chunks
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import base64
from pathlib import Path
import time



st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Helper function to load and encode images
def get_image_as_base64(file_path):
    """Loads an image file and returns it as a base64 encoded string."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Image file not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading image {file_path}: {e}")
        return None

YOUTUBE_LOGO_PATH = "image/youtube.png"
BOT_LOGO_PATH = "image/bot.png"
youtube_exists = Path(YOUTUBE_LOGO_PATH).is_file()
bot_exists = Path(BOT_LOGO_PATH).is_file()
youtube_b64 = get_image_as_base64(YOUTUBE_LOGO_PATH)
bot_b64 = get_image_as_base64(BOT_LOGO_PATH)

if youtube_b64 and bot_b64:
    # Create the base64 image source strings
    youtube_src = f"data:image/png;base64,{youtube_b64}"
    bot_src = f"data:image/png;base64,{bot_b64}"

    st.markdown(f"""
    <div style="display: flex; align-items: stretch; justify-content: left; width: 100%;">
        <img src="{youtube_src}" width="80" style="margin-right: 20px;">
        <h1 style="margin: 0; white-space: nowrap;">YouTube Video Chatbot</h1>
        <img src="{bot_src}" width="70" style="margin-left: 20px;">
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("Could not find or encode 'youtube.png' or 'bot.png'")



# --- 2. RAG Chain Logic ---
@st.cache_resource  
def get_base_components():
    """Gets the (expensive) components that don't change."""
    prompt_template = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer the question based only on the following context.
        If you cannot find the answer in the context, just say "I don't know".

        Context:
        {context}

        Question:
        {query}
        """,
        input_variables=['context', 'query']
    )
    llm = ChatOpenAI(model='gpt-4.1-nano')
    parser = StrOutputParser()
    return prompt_template, llm, parser

def create_query_chain(vector_store):
    """
    Creates the "query" part of the RAG chain.
    This takes an existing, in-memory vector store.
    """
    prompt, llm, parser = get_base_components()

    retriever = vector_store.as_retriever()

    # Define the chain that queries this retriever
    query_chain = (
        RunnableParallel({"context": retriever | RunnableLambda(format_doc), "query": RunnablePassthrough()})
        | prompt
        | llm
        | parser
    )
    return query_chain

# --- 3. Callback functions for clearing ---

def clear_chat_data():
    """Clears just the chat messages and the cached vector store."""
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Enter a YouTube URL to get started."}]
    st.session_state.vector_store = None 
    st.session_state.current_url = None

def clear_chat_and_url():
    """Clears all chat data AND the URL input box."""
    clear_chat_data()
    if 'url_input' in st.session_state:
        st.session_state.url_input = ""

# --- 4. Sidebar for Inputs and Controls ---
with st.sidebar:
    st.header("Controls")
    
    st.text_input(
        "Enter YouTube URL:", 
        key="url_input",
        placeholder="Enter a URL."
    )
    
    if st.button("Process Video"):
        if st.session_state.url_input:
            
            progress_bar = st.progress(0, text="Starting...")
            
            try:
                # Clear any old data
                clear_chat_data() 
                st.session_state.current_url = st.session_state.url_input
                
                # 1. Get Transcript
                progress_bar.progress(25, text="Fetching transcript...")
                transcript_text = get_transcript_from_url(st.session_state.url_input)
                
                if "Failed" in transcript_text or "Error" in transcript_text or "disabled" in transcript_text:
                    st.error(f"Error: {transcript_text}")
                    progress_bar.empty() # Clear the progress bar on error
                else:
                    # 2. Split Transcript
                    progress_bar.progress(50, text="Splitting transcript...")
                    chunks = split_transcript(transcript_text)
                    
                    # 3. Create Vector Store (This is the slowest part)
                    progress_bar.progress(75, text="Creating vector store (this may take a moment)...")
                    vector_store = vectorstore_from_chunks(chunks)
                    
                    # 4. Store the vector store in session state
                    st.session_state.vector_store = vector_store
                    
                    progress_bar.progress(100, text="Processing complete!")
                    st.success("Video processed! You can now ask questions.")
                    st.session_state.messages = [{"role": "assistant", "content": "I've processed the video. Ask me anything about it!"}]
                    
                    time.sleep(1)
                    progress_bar.empty() 

            except Exception as e:
                st.error(f"An error occurred: {e}")
                progress_bar.empty() 
        else:
            st.warning("Please enter a YouTube URL.")

    st.button("Clear Chat", on_click=clear_chat_and_url, type="primary")


# --- 5. Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Enter a YouTube URL to get started."}]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None 


# --- 6. Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- 7. Chat Input ---
if prompt := st.chat_input("Ask a question about the video..."):
    if st.session_state.vector_store is None:
        st.warning("Please process a YouTube video first.", icon="⚠️")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    query_chain = create_query_chain(st.session_state.vector_store)
                    response = query_chain.invoke(prompt)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")





        