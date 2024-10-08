# create the models that will define our database connections.

from sqlalchemy import Column, ForeignKey, Integer, String, TIMESTAMP, text
import datetime
from database import Base
from sqlalchemy.orm import relationship

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, nullable=False)
    title = Column(String, nullable=False)
    image = Column(String, nullable=True)
    type = Column(String, nullable=True)
    url = Column(String, nullable=True)
    index_path = Column(String, unique=True, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=True, server_default=text('Now()'))

    chats = relationship("Chat", back_populates="subject")

class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    question = Column(String)
    response = Column(String)

    subject = relationship("Document", back_populates="chats")