from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Annotated
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session

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

@app.post("/docs/") # the ending slash is important!
async def create_document(document: DocumentBase, db: db_dependency):
    db_document = models.Documents(title=document.title, image=document.image, type=document.type, url=document.url)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

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