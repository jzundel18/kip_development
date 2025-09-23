# models.py - Keep only the models
import pickle
import hashlib
from typing import Optional
from sqlmodel import SQLModel, Field
from openai import OpenAI


class SolicitationRaw(SQLModel, table=True):
    __tablename__ = "solicitationraw"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    pulled_at: Optional[str] = Field(default=None, index=True)
    notice_id: str = Field(index=True, nullable=False, unique=True)
    solicitation_number: Optional[str] = None
    title: Optional[str] = None
    notice_type: Optional[str] = None
    posted_date: Optional[str] = Field(default=None, index=True)
    response_date: Optional[str] = Field(default=None, index=True)
    archive_date: Optional[str] = Field(default=None, index=True)
    naics_code: Optional[str] = Field(default=None, index=True)
    set_aside_code: Optional[str] = Field(default=None, index=True)
    description: Optional[str] = None
    link: Optional[str] = None
    # Add the new POP fields
    pop_city: Optional[str] = None
    pop_state: Optional[str] = None
    pop_zip: Optional[str] = None
    pop_country: Optional[str] = None
    pop_raw: Optional[str] = None


class User(SQLModel, table=True):
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True, nullable=False)
    password_hash: str = Field(nullable=False)
    created_at: Optional[str] = Field(default=None)


class CompanyProfile(SQLModel, table=True):
    __tablename__ = "company_profile"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, nullable=False)
    company_name: str = Field(nullable=False)
    description: str = Field(nullable=False)
    city: Optional[str] = None
    state: Optional[str] = None
    created_at: Optional[str] = Field(default=None)
    updated_at: Optional[str] = Field(default=None)


class Company(SQLModel, table=True):
    __tablename__ = "company"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = Field(default=None, index=True)


class SolicitationEmbedding(SQLModel, table=True):
    __tablename__ = "solicitation_embeddings"
    __table_args__ = {"extend_existing": True}
    notice_id: str = Field(primary_key=True, index=True)
    embedding: str = Field()  # JSON-encoded numpy array
    text_hash: str = Field(index=True)  # Hash of title+description
    created_at: str = Field(default=None)
