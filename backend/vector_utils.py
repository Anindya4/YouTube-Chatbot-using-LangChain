from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()

def vectorstore_from_chunks(chunks: List[Dict]) -> FAISS:
    """
    Create FAISS vectorstore from given chunk list
    Args:
        chunks: List[Dict] - List of chunks for creating vectorstore
    Returns:
        FAISS document obj
    """
    
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding
        )
    return vectorstore


def vectorstore_for_query(query:str, vectorstore:FAISS):
    """Return the vectorstore for given query"""
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    retriever_doc = retriever.invoke(query)
    return retriever_doc
