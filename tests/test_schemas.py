"""Tests du contrat des schemas d'API (dependances legeres : pydantic seul)."""

from app.api.schemas import AskRequest, AskResponse, ModelInfo, ModelsResponse


def test_ask_request_model_is_optional():
    assert AskRequest(question="q").model is None
    assert AskRequest(question="q", model="llama3.2:1b").model == "llama3.2:1b"


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
