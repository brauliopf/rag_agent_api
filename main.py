from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Annotated
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session

from llama_index.core import(
    VectorStoreIndex, SimpleDirectoryReader,
    StorageContext, load_index_from_storage
)

app = FastAPI()
models.Base.metadata.create_all(bind=engine) # create all tables and columns in Postgres

# Define the classes. Declare relationships as parts of the classes. For example:
# Questions: text + choices | eventhough choice is not part of the question table, it is part of the question class
class DocumentBase(BaseModel):
    title: str
    image: str
    type: str
    url: str

class ChatBase(BaseModel):
    document: DocumentBase
    question: str
    response: str

def getdb():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(getdb)]

# get index from llama_index
import urllib.request

def download_file(file_url: str):
    # download file
    try:
        download_result, headers = urllib.request.urlretrieve(url=file_url)
    except Exception as e:
        print("Error downloading file:", e)
        return {"error": str(e)}

    # check download success
    if isinstance(download_result, dict) and "error" in download_result:
        print("Download failed:", download_result["error"])
    else:
        print("File downloaded successfully")

    return download_result

def generate_index(doc_id:int, file_path: str):

    # parse file
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    # create index
    index = VectorStoreIndex.from_documents(documents, insert_batch_size=2048)
    
    # persist store in local
    INDEX_STORE = f'./store/{doc_id}'
    print('INDEX_STORE:', INDEX_STORE)

    index.storage_context.persist(INDEX_STORE)

    return {'index': index, 'index_path': INDEX_STORE}

import os
from llama_parse import LlamaParse
import nest_asyncio; nest_asyncio.apply()
def generate_index_llama_parse(doc_id:int, file_path: str):

    LLAMA_CLOUD_API_KEY='llx-5GA1VVuB12niXoT5rV5AzNHFsLKWwPTAH62EYe9U8IJCA1ZV'
    documents = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown").load_data(file_path)

    # create index
    index = VectorStoreIndex.from_documents(documents, insert_batch_size=2048)
    
    # persist store in local
    INDEX_STORE = f'./store/{doc_id}'
    print('INDEX_STORE:', INDEX_STORE)

    index.storage_context.persist(INDEX_STORE)

    return {'index': index, 'index_path': INDEX_STORE}

def load_index_from_storage(index_store: str):
    # load index from storage
    storage_context = StorageContext.from_defaults(persist_dir=index_store)
    index = load_index_from_storage(storage_context)
    return index

@app.post("/docs/") # the ending slash is important!
async def create_document(document: DocumentBase, db: db_dependency):
    db_document = models.Documents(title=document.title, image=document.image, type=document.type, url=document.url)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    # download document
    file_path = download_file(db_document.url)

    # generate index
    index, index_path = generate_index(doc_id=db_document.id, file_path=file_path).values()
    print(index)

    # query the document
    query_engine = index.as_query_engine()
    response = query_engine.query('How much Revenue did Disney earned in 1992 with Consumer Products?')

    print(response)
    
    return response

@app.get("/docs/")
async def list_document(db: db_dependency):
    result =  db.query(models.Documents).all()
    if not result:
        raise HTTPException(status_code=404, detail="No documents found")
    return

@app.get("/docs/{document_id}/summary")
async def document_summary(document_id: int, db: db_dependency):
    result = db.query(models.Documents).filter(models.Documents.id == document_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # read doc content
    

    # ask llm to sumaarize
    # return summary

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/testdb")
async def testdb():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT * FROM public.sandbox LIMIT 3"))
            rows = [dict(row) for row in result]
            
        # Print rows
        for row in rows:
            print(row)
        
    except Exception as e:
        print('Error:', e)
        return {"error": str(e)}
    
    # Return the list of rows
    return rows


@app.post("/docs")
async def upload_doc():

    return {"message": "Hello World"}