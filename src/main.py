from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from typing import List, Annotated, Literal
from sqlalchemy.orm import Session
from llama_index.core import(
    VectorStoreIndex, SimpleDirectoryReader,
    StorageContext, load_index_from_storage
)
from llama_index.core import Document
from references.download_from_google_drive import download_google_drive

## query engine
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SummaryIndex, VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext

## send http requests
import urllib.request
## engage a db
import models
import orm
from database import engine, SessionLocal

# LOAD DATABASE
## map ORM models
DocumentBase = orm.DocumentBase
ChatBase = orm.ChatBase

## create all tables and columns
models.Base.metadata.create_all(bind=engine)

## DB CONNECT
def getdb():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
db_dependency = Annotated[Session, Depends(getdb)]

# AUXILIARY FUNCTIONS
def get_index_from_nodes(type: Literal['SUM', 'VEC'], nodes: List[Document]):
    """
    Get index from nodes.
    type: SUM for SummaryIndex, VEC for VectorStoreIndex.
    """
    if type == 'SUM':
        index = SummaryIndex(nodes)
    elif type == 'VEC':
        index = VectorStoreIndex(nodes)
    return index

async def get_router_query_engine(documents,
                                  llm = OpenAI(model="gpt-3.5-turbo"),
                                  embed_model = OpenAIEmbedding(model="text-embedding-ada-002")):
    """
    Get router query engine.
    documents is a list of llama_index.core.Document
    Default llm and embed_model are OpenAI models.
    """
    
    # execute splitting into nodes (chunks)
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # set up llm and embeddings generator
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # create index vectors: for summary and query engines
    summary_index = get_index_from_nodes(type = 'SUM', nodes = nodes)
    vector_index = get_index_from_nodes(type = 'VEC', nodes = nodes)
    
    if os.path.exists("../index/summary_index"):
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir="../index/summary_index")
        summary_index = load_index_from_storage(storage_context)
    else:
        summary_index.storage_context.persist("../index/summary_index")
    if os.path.exists("../index/vector_index"):
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir="../index/vector_index")
        vector_index = load_index_from_storage(storage_context)
    else:
        vector_index.storage_context.persist("../index/vector_index")

    # build tools for the query engine (index as engine) + metadata
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        ),
        description=(
            "Useful for summarization questions. Use this to answer exploratory questions about the topic, while keeping limited to the content from the document provided, not from external sources."
        ),
    )
    
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(),
        description=(
            "Useful for retrieving specific context from the document shared."
        ),
    )
    
    # build query engine
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )
    return query_engine

# DEFINE ROUTES
app = FastAPI()

import tempfile
import os, io
import shutil
from PyPDF2 import PdfReader

@app.post("/docs/")
async def create_document(file: UploadFile = File(...)):
    
    # create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)
        print('TEMP_DIR', temp_dir, )

        try:
            # read the file contents
            contents = file.file.read()
            # create file with the same name inside the temporary directory
            with open(temp_file_path, "wb") as temp_file:
                # and write the contents to it
                temp_file.write(contents)
        except Exception:
            return {"message": "There was an error uploading the file"}
        
        try:
            # Process the file
            documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
            query_engine = await get_router_query_engine(documents,
                                                        #  llm = Groq("llama-3.1-70b-versatile")
                                                         )
        except Exception:
            return {"message": "There was an error generating the query engine"}
        finally:
            file.file.close()
            # The file will be automatically deleted when exiting this block

    response = query_engine.query("Please summarize this document")
    
    return response

# Uses file downloaded from Google Drive! Success.
@app.get("/docs/")
async def get_doc(llm=OpenAI(model="gpt-3.5-turbo")):

    # use GoogleDriveReader
    # https://docs.llamaindex.ai/en/stable/examples/data_connectors/GoogleDriveDemo/

    file_path = "../assets/birth_preferences.pdf"
    document = SimpleDirectoryReader(input_files=[file_path]).load_data()
    llm = Groq("llama-3.1-70b-versatile")
    query_engine = await get_router_query_engine(document, llm = llm)
    response = query_engine.query("Please summarize this document")

    return {'response': response, 'model': llm}

@app.post("/docs/gdrive/")
def create_doc(id: str, filename: str):
    response = download_google_drive(id, filename)
    return {"message": "Document downloaded successfully.", "filepath":response}