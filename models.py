from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from database import Base

class Documents(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    title = Column(String)
    url = Column(String)
    type = Column(String)
    image = Column(String)

class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    question = Column(String)
    response = Column(String)