"""Modeles SQLAlchemy : notebooks, messages de chat, notes."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Notebook(Base):
    """Un notebook = un workspace, avec un titre libre distinct de l'id (slug sur)."""

    __tablename__ = "notebooks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # slug (= workspace)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="notebook",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )
    notes: Mapped[list["Note"]] = relationship(
        back_populates="notebook",
        cascade="all, delete-orphan",
        order_by="Note.id",
    )


class Message(Base):
    """Un tour de conversation (question utilisateur ou reponse assistant)."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    notebook_id: Mapped[str] = mapped_column(
        ForeignKey("notebooks.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Pour un message assistant : sources citees + numeros cites + modele (JSON).
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    cited: Mapped[list | None] = mapped_column(JSON, nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    notebook: Mapped[Notebook] = relationship(back_populates="messages")


class Note(Base):
    """Une note libre du panneau Studio, rattachee a un notebook."""

    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    notebook_id: Mapped[str] = mapped_column(
        ForeignKey("notebooks.id", ondelete="CASCADE"), index=True, nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    notebook: Mapped[Notebook] = relationship(back_populates="notes")
