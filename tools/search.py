from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings 
import pathlib
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os
from langchain_community.document_loaders.text import TextLoader 
from uuid import uuid4 
from dotenv import load_dotenv
import json
from langchain.schema import Document
load_dotenv()


def get_vectorstore() -> Chroma:
    #Check if vector store exists and initialize if needed
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore_path = pathlib.Path("./chroma_langchain_db")
    if not vectorstore_path.exists() or not any (vectorstore_path.iterdir()):
        initialize_rag()
    return Chroma(
        embedding_function=embeddings, 
        persist_directory="./chroma_langchain_db",
    )

# def initialize_rag():
#     #Initialize the RAG system by creating and populating the vector store.
#     embeddings = OpenAIEmbeddings (model="text-embedding-3-large" )
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000, 
#         chunk_overlap=200, 
#         length_function=len,
#     )
#     documents = []
#     docs_dir = "./docs"
#     print(f"Loading from {docs_dir}... found {len(os.listdir(docs_dir))} files")

#     if os.path.exists(docs_dir):
#         for filename in os.listdir(docs_dir):
#             if filename.endswith(".txt"):
#                 file_path = os.path.join(docs_dir, filename)
#                 loader = TextLoader(file_path)
#                 loaded_docs = loader.load()
#                 texts = text_splitter.split_documents(loaded_docs)
#                 documents.extend(texts)

#     uuids = [str(uuid4()) for _ in range(len(documents))]
#     vectorstore = Chroma.from_documents(
#         documents=documents, 
#         embedding=embeddings, 
#         ids=uuids, 
#         persist_directory="./chroma_langchain_db",
#     )
#     return vectorstore


# def initialize_rag():
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#     )
#     documents = []
#     docs_dir = "./docs"

#     if os.path.exists(docs_dir):
#         for filename in os.listdir(docs_dir):
#             file_path = os.path.join(docs_dir, filename)

#             # Case 1: plain text files
#             if filename.endswith(".txt"):
#                 loader = TextLoader(file_path)
#                 loaded_docs = loader.load()
#                 texts = text_splitter.split_documents(loaded_docs)
#                 documents.extend(texts)

#             # Case 2: Slack JSON threads
#             elif filename.endswith(".json"):
#                 with open(file_path, "r") as f:
#                     data = json.load(f)

#                 # dataset is a list of threads
#                 for thread in data:
#                     thread_ts = thread.get("thread_ts", "")
#                     channel = thread.get("channel", "unknown")

#                     for msg in thread.get("messages", []):
#                         user = msg.get("user", "unknown")
#                         text = msg.get("text", "")
#                         ts = msg.get("ts", "")

#                         # Flatten into natural text for embeddings
#                         content = f"[{channel} | {thread_ts}] User {user} said: {text}"

#                         doc = Document(
#                             page_content=content,
#                             metadata={
#                                 "thread_ts": thread_ts,
#                                 "channel": channel,
#                                 "user": user,
#                                 "ts": ts
#                             }
#                         )
#                         documents.append(doc)

#     # Split all documents (txt + json-based) for embeddings
#     docs_to_index = text_splitter.split_documents(documents)

#     uuids = [str(uuid4()) for _ in range(len(docs_to_index))]
#     vectorstore = Chroma.from_documents(
#         documents=docs_to_index,
#         embedding=embeddings,
#         ids=uuids,
#         persist_directory="./chroma_langchain_db",
#     )
#     return vectorstore



def initialize_rag():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    documents = []
    docs_dir = "./knowledge_base"

    if os.path.exists(docs_dir):
        for filename in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, filename)

            # Plain text files
            if filename.endswith(".txt"):
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                texts = text_splitter.split_documents(loaded_docs)
                documents.extend(texts)

            # Slack JSON threads
            elif filename.endswith(".json"):
                with open(file_path, "r") as f:
                    threads = json.load(f)

                for thread in threads:
                    thread_ts = thread.get("thread_ts", "")
                    channel = thread.get("channel", "unknown")
                    for msg in thread.get("messages", []):
                        user = msg.get("user", "unknown")
                        text = msg.get("text", "")
                        ts = msg.get("ts", "")

                        # Flatten message with thread info for easy retrieval
                        content = (
                            f"[Channel: {channel} | Thread: {thread_ts}] "
                            f"User {user} said: {text}"
                        )

                        doc = Document(
                            page_content=content,
                            metadata={
                                "thread_ts": thread_ts,
                                "channel": channel,
                                "user": user,
                                "ts": ts
                            }
                        )
                        documents.append(doc)

    # Split all documents for embeddings
    docs_to_index = text_splitter.split_documents(documents)

    uuids = [str(uuid4()) for _ in range(len(docs_to_index))]
    vectorstore = Chroma.from_documents(
        documents=docs_to_index,
        embedding=embeddings,
        ids=uuids,
        persist_directory="./chroma_langchain_db",
    )
    return vectorstore