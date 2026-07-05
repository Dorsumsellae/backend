"""Acces aux donnees applicatives (notebooks, messages, notes).

Fonctions parametrees par une `Session` SQLAlchemy (pas de session globale), pour
rester testables avec un moteur SQLite en memoire.
"""

from sqlalchemy import delete, select

from app.db.models import Message, Note, Notebook
from app.notebooks import slugify


# --- Notebooks --------------------------------------------------------------


def list_notebooks(session) -> list[Notebook]:
    """Tous les notebooks, tries par titre."""
    return list(session.scalars(select(Notebook).order_by(Notebook.title)))


def get_notebook(session, notebook_id: str) -> Notebook | None:
    return session.get(Notebook, notebook_id)


def get_or_create_notebook(session, notebook_id: str, title: str | None = None) -> Notebook:
    """Retourne le notebook, en le creant (title = id par defaut) s'il n'existe pas."""
    notebook = session.get(Notebook, notebook_id)
    if notebook is None:
        notebook = Notebook(id=notebook_id, title=title or notebook_id)
        session.add(notebook)
        session.flush()
    return notebook


def create_notebook(session, title: str) -> Notebook:
    """Cree un notebook a partir d'un titre libre ; l'id est un slug unique."""
    base = slugify(title)
    notebook_id = base
    suffix = 2
    while session.get(Notebook, notebook_id) is not None:
        notebook_id = f"{base}-{suffix}"
        suffix += 1
    notebook = Notebook(id=notebook_id, title=(title or "").strip() or notebook_id)
    session.add(notebook)
    session.flush()
    return notebook


def rename_notebook(session, notebook_id: str, title: str) -> Notebook | None:
    notebook = session.get(Notebook, notebook_id)
    if notebook is None:
        return None
    notebook.title = (title or "").strip() or notebook.title
    session.flush()
    return notebook


def delete_notebook(session, notebook_id: str) -> bool:
    """Supprime le notebook et, par cascade, ses messages et notes."""
    notebook = session.get(Notebook, notebook_id)
    if notebook is None:
        return False
    session.delete(notebook)
    return True


def backfill_notebooks(session, workspace_ids) -> None:
    """Cree une ligne notebook (title = id) pour chaque workspace connu sans entree."""
    existing = set(session.scalars(select(Notebook.id)))
    for workspace_id in workspace_ids:
        if workspace_id and workspace_id not in existing:
            session.add(Notebook(id=workspace_id, title=workspace_id))
            existing.add(workspace_id)


# --- Messages ---------------------------------------------------------------


def list_messages(session, notebook_id: str) -> list[Message]:
    return list(
        session.scalars(
            select(Message)
            .where(Message.notebook_id == notebook_id)
            .order_by(Message.id)
        )
    )


def add_message(
    session,
    notebook_id: str,
    role: str,
    content: str,
    sources=None,
    cited=None,
    model=None,
) -> Message:
    message = Message(
        notebook_id=notebook_id,
        role=role,
        content=content,
        sources=sources,
        cited=cited,
        model=model,
    )
    session.add(message)
    session.flush()
    return message


def clear_messages(session, notebook_id: str) -> int:
    result = session.execute(
        delete(Message).where(Message.notebook_id == notebook_id)
    )
    return result.rowcount or 0


# --- Notes ------------------------------------------------------------------


def list_notes(session, notebook_id: str) -> list[Note]:
    return list(
        session.scalars(
            select(Note).where(Note.notebook_id == notebook_id).order_by(Note.id)
        )
    )


def add_note(session, notebook_id: str, text: str) -> Note:
    note = Note(notebook_id=notebook_id, text=text)
    session.add(note)
    session.flush()
    return note


def delete_note(session, notebook_id: str, note_id: int) -> bool:
    note = session.get(Note, note_id)
    if note is None or note.notebook_id != notebook_id:
        return False
    session.delete(note)
    return True
