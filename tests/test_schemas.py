"""Tests du contrat des schemas d'API (dependances legeres : pydantic seul)."""

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ModelsResponse,
    Source,
)


def test_ask_request_model_is_optional():
    assert AskRequest(question="q").model is None
    assert AskRequest(question="q", model="llama3.2:1b").model == "llama3.2:1b"


def test_ask_request_accepts_filenames():
    assert AskRequest(question="q").filenames is None
    assert AskRequest(question="q", filenames=["a.txt", "b.txt"]).filenames == [
        "a.txt",
        "b.txt",
    ]


def test_source_carries_cite_index():
    assert Source(filename="a.txt", passage_id=0, excerpt="x").cite is None
    assert Source(filename="a.txt", passage_id=0, excerpt="x", cite=2).cite == 2


def test_chat_request_roundtrip():
    req = ChatRequest(
        messages=[{"role": "user", "content": "salut"}], filenames=["a.txt"]
    )
    assert req.messages[-1].role == "user"
    assert req.filenames == ["a.txt"]


def test_chat_message_rejects_unknown_role():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "system", "content": "x"}])


def test_chat_response_shape():
    resp = ChatResponse(answer="a", sources=[], model="qwen2.5:0.5b", cited=[1])
    assert resp.model == "qwen2.5:0.5b"
    assert resp.cited == [1]


def test_ask_response_exposes_model():
    resp = AskResponse(question="q", answer="a", sources=[], model="qwen2.5:0.5b")
    assert resp.model == "qwen2.5:0.5b"


def test_models_response_shape():
    resp = ModelsResponse(
        models=[ModelInfo(name="qwen2.5:0.5b", is_default=True)],
        default="qwen2.5:0.5b",
    )
    assert resp.models[0].is_default is True
    assert resp.default == "qwen2.5:0.5b"
