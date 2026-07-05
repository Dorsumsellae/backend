"""Tests de la route conversationnelle /chat (pipeline LLM/vectorstore stubbe)."""

from fastapi.testclient import TestClient

from app.rag import pipeline
from main import app

client = TestClient(app)


def test_chat_requires_last_message_to_be_user():
    resp = client.post(
        "/chat", json={"messages": [{"role": "assistant", "content": "bonjour"}]}
    )
    assert resp.status_code == 400


def test_chat_requires_non_empty_messages():
    resp = client.post("/chat", json={"messages": []})
    assert resp.status_code == 400


def test_chat_returns_answer_and_sources(monkeypatch):
    def fake_answer_chat(messages, workspace, top_k=None, model=None, filenames=None):
        # On verifie au passage que la route transmet bien le workspace et le sous-ensemble.
        assert workspace == "alpha"
        assert filenames == ["a.txt"]
        return {
            "answer": "reponse [1]",
            "sources": [
                {"filename": "a.txt", "passage_id": 0, "excerpt": "extrait", "cite": 1}
            ],
            "model": "qwen2.5:0.5b",
            "cited": [1],
        }

    monkeypatch.setattr(pipeline, "answer_chat", fake_answer_chat)

    resp = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "premiere"},
                {"role": "assistant", "content": "reponse"},
                {"role": "user", "content": "seconde"},
            ],
            "workspace": "alpha",
            "filenames": ["a.txt"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "reponse [1]"
    assert data["model"] == "qwen2.5:0.5b"
    assert data["cited"] == [1]
    assert data["sources"][0]["filename"] == "a.txt"
    assert data["sources"][0]["cite"] == 1
