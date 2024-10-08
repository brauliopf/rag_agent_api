from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from typing import List, Annotated, Literal, Optional
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
from models import Document as SQLAlchemyDocument
import schemas as schemas
from database import engine, SessionLocal

# LOAD DATABASE
## map ORM models
DocumentBase = schemas.DocumentBase
ChatBase = schemas.ChatBase

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
def get_documents(document):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, document.filename)

            try:
                # read file contents
                contents = document.file.read()
                # create file with the same name inside the temporary directory
                with open(temp_file_path, "wb") as temp_file:
                    # and write the contents to it
                    temp_file.write(contents)
            except Exception:
                return {"message": "There was an error uploading the file"}
            
            try:
                # read the file
                documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
            except Exception:
                return {"message": "There was an error generating the query engine"}
            finally:
                document.file.close()
                # The file will be automatically deleted when exiting this block
            
            return {"documents": documents, "metadata":{"filename": document.filename, "content_type": document.content_type}}

def persist_index(index, path):
    
    index.storage_context.persist(path)
    pass

def load_index(path):
    storage_context = StorageContext.from_defaults(persist_dir=path)
    return load_index_from_storage(storage_context)

def get_index(type: Literal['SUM', 'VEC'], documents: List[Document], document_id: Optional[str] = None):
    # function expects a list (as SimpleDirectoryReader would return),

    # execute splitting into nodes (chunks)
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = splitter.get_nodes_from_documents(documents)

    if type == 'SUM':
        if os.path.exists("../index/summary_index"):
            index = load_index("../index/summary_index")
        else:
            # get index from nodes + persist
            index = SummaryIndex(nodes)
            persist_index(index, "../index/summary_index")
    elif type == 'VEC':
        if os.path.exists("../index/vector_index"):
            index = load_index("../index/vector_index")
        else:
            # get index from nodes + persist
            index = VectorStoreIndex(nodes)
            persist_index(index, "../index/vector_index")

    return index

def get_index_with_id(type: Literal['SUM', 'VEC'], documents: List[Document], document_id: int):
    # function expects a list (as SimpleDirectoryReader would return),

    # execute splitting into nodes (chunks)
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = splitter.get_nodes_from_documents(documents)

    if type == 'SUM':
        if os.path.exists(f'../index/{document_id}/summary_index'):
            index = load_index(f'../index/{document_id}/summary_index')
        else:
            # get index from nodes + persist
            index = SummaryIndex(nodes)
            persist_index(index, f'../index/{document_id}/summary_index')
    elif type == 'VEC':
        if os.path.exists(f'../index/{document_id}/vector_index'):
            index = load_index(f'../index/{document_id}/vector_index')
        else:
            # get index from nodes + persist
            index = VectorStoreIndex(nodes)
            persist_index(index, f'../index/{document_id}/vector_index')

    return index

async def get_router_query_engine(summary_index, vector_index,
                                  llm: OpenAI = OpenAI(model="gpt-3.5-turbo"),
                                  embed_model: OpenAIEmbedding = OpenAIEmbedding(model="text-embedding-ada-002")):
    """
    Get router query engine.
    documents is a list of llama_index.core.Document
    Default llm and embed_model are OpenAI models.
    """
    
    # set up llm and embeddings generator
    Settings.llm = llm
    Settings.embed_model = embed_model

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
import os
from PyPDF2 import PdfReader

@app.post("/docs/")
async def create_document(file: UploadFile = File(...), db: Session = Depends(getdb)):
    '''
    Create a document from an uploaded file: title, type, index_path.
    Returns the document (with id).
    '''
    
    # generate parseable documents from uploaded file
    documents = get_documents(file)
    # documents['metadata']['filename']

    # persist document
    print("****", documents['metadata']['filename'], documents['metadata']['content_type'])

    db_document = SQLAlchemyDocument(
        title=documents['metadata']['filename'],
        type=documents['metadata']['content_type'])
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    db.close()
    
    # create index: for summary and query (vector) engines
    summary_index = get_index_with_id(type='SUM', documents=documents['documents'], document_id=db_document.id)
    vector_index = get_index_with_id(type='VEC', documents=documents['documents'], document_id=db_document.id)

    # generate query engine
    query_engine = await get_router_query_engine(
        summary_index, vector_index,
        #  llm = Groq("llama-3.1-70b-versatile")
        )
            
    response = query_engine.query("Please summarize the document")
    
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
