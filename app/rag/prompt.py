"""Construction du prompt enrichi envoye au LLM."""

PROMPT_TEMPLATE = """Tu es un assistant documentaire.
Reponds uniquement a partir du contexte fourni.
Si l'information n'est pas presente dans le contexte, reponds :
"Je ne trouve pas cette information dans le document fourni."

Contexte :
{context}

Question :
{question}

Reponse :"""


def build_prompt(question: str, passages: list[str]) -> str:
    """Assemble le prompt final a partir de la question et des passages retrouves.

    Args:
        question: question de l'utilisateur.
        passages: liste des extraits de texte retrouves dans la base vectorielle.

    Returns:
        Le prompt complet, pret a etre envoye au LLM.
    """
    context = "\n\n".join(
        f"[Passage {i + 1}]\n{p.strip()}" for i, p in enumerate(passages)
    )
    return PROMPT_TEMPLATE.format(context=context, question=question.strip())
