# app/models.py

from sqlalchemy import Column, Integer, String
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    number = Column(String, unique=True, index=True)
    usertype = Column(String)
    name = Column(String)
    password = Column(String)
    embed = Column(String)
