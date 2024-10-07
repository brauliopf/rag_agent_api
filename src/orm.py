from pydantic import BaseModel
from typing import Optional

# Define the classes. Declare relationships as parts of the classes. For example:
# Questions: text + choices | eventhough choice is not part of the question table, it is part of the question class
class DocumentBase(BaseModel):
    title: str
    image: str
    type: str
    url: str
    query: Optional[str] = None

class ChatBase(BaseModel):
    document: DocumentBase
    question: str
    response: str

