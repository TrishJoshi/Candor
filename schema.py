"""
schema.py — Universal Data Contract for Project Candor.
All modules must import from here. Never redefine these models elsewhere.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


# ── Facebook source schema (ingest layer) ─────────────────────────────────────

class Author(BaseModel):
    id: Optional[str] = None
    name: str
    profile_url: Optional[str] = None


class Reactions(BaseModel):
    total: int
    like: Optional[int] = None
    love: Optional[int] = None
    care: Optional[int] = None
    haha: Optional[int] = None
    wow: Optional[int] = None
    sad: Optional[int] = None
    angry: Optional[int] = None


class Attachment(BaseModel):
    type: str
    url: Optional[str] = None
    title: Optional[str] = None


class FacebookComment(BaseModel):
    """Raw Facebook comment as extracted by the scraper / ingest layer."""
    comment_id: str
    author: Author
    message: Optional[str] = ""
    attachments: List[Attachment] = []
    reaction_count: int
    reactions: Optional[Reactions] = None
    created_time: Optional[str] = None
    replies: List["FacebookComment"] = []


class FacebookPost(BaseModel):
    """Top-level Facebook post object matching Facebook_JSON_Schema.json."""
    post_id: str
    permalink_url: Optional[str] = None
    author: Author
    source: dict
    message: Optional[str] = ""
    attachments: List[Attachment] = []
    reactions: Reactions
    share_count: Optional[int] = None
    comment_count: Optional[int] = None
    created_time: Optional[str] = None
    comments: List[FacebookComment] = []


# ── Candor pipeline schema (tracks 2-5) ───────────────────────────────────────

class Comment(BaseModel):
    """Normalised comment that flows through the Candor pipeline."""
    comment_id: str
    author: str
    account_age_days: int = 0           # populated by Track 1 when available
    timestamp: float                     # unix epoch
    text: str
    cluster_id: Optional[int] = None    # populated by Track 2
    account_authenticity_score: Optional[float] = None  # populated by Track 3


class NarrativeCluster(BaseModel):
    cluster_id: int
    label: Optional[str] = None         # populated by Track 4
    cluster_score: Optional[float] = None  # populated by Track 3
    comments: List[Comment]
