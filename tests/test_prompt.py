"""Tests de la construction du prompt."""

from app.rag.prompt import build_prompt


def test_prompt_contains_question_and_context():
    prompt = build_prompt("Que dit le texte ?", ["Passage A", "Passage B"])
    assert "Que dit le texte ?" in prompt
    assert "Passage A" in prompt
    assert "Passage B" in prompt


def test_prompt_contains_fallback_instruction():
    prompt = build_prompt("Question", ["contexte"])
    assert "Je ne trouve pas cette information dans le document fourni." in prompt
