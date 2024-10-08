from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# Define the classes. Declare relationships as parts of the classes. For example:
# Questions: text + choices | eventhough choice is not part of the question table, it is part of the question class
class DocumentBase(BaseModel):
    title: str
    image: Optional[str] = None
    type: str
    url: Optional[str] = None
    index_path: Optional[str] = None
    created_at: Optional[str] = None
    query: Optional[str] = None # user query. used in the application only

class ChatBase(BaseModel):
    document: DocumentBase
    question: str
    response: str

