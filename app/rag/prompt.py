"""Construction des prompts enrichis envoyes au LLM (RAG mono-tour et chat)."""

import re

# Consigne commune : reponse ancree dans le contexte + citations numerotees.
# Le numero cite `[n]` correspond exactement au numero du passage dans le contexte
# (voir `_format_context`), ce qui maximise les chances qu'un petit modele cite bien.
_CITATION_RULES = """Reponds uniquement a partir du contexte numerote ci-dessous.
Apres chaque affirmation, cite le ou les passages utilises avec leur numero
entre crochets, par exemple [1] ou [2][3]. N'invente aucune source.
Si l'information n'est pas presente dans le contexte, reponds :
"Je ne trouve pas cette information dans le document fourni." """

PROMPT_TEMPLATE = """Tu es un assistant documentaire.
{rules}

Contexte :
{context}

Question :
{question}

Reponse :"""

CHAT_PROMPT_TEMPLATE = """Tu es un assistant documentaire conversationnel.
{rules}

Contexte :
{context}

Conversation :
{history}"""

# Nombre de tours de conversation conserves dans le prompt de chat (fenetre glissante).
_HISTORY_TURNS = 6


def _format_context(passages: list[str]) -> str:
    """Numerote les passages `[1] ...` (le numero sert d'ancre de citation)."""
    return "\n\n".join(f"[{i + 1}] {p.strip()}" for i, p in enumerate(passages))


def build_prompt(question: str, passages: list[str]) -> str:
    """Assemble le prompt RAG mono-tour (consigne + contexte numerote + question)."""
    return PROMPT_TEMPLATE.format(
        rules=_CITATION_RULES,
        context=_format_context(passages),
        question=question.strip(),
    )


def build_chat_prompt(messages: list[dict], passages: list[str]) -> str:
    """Assemble le prompt de chat multi-tours.

    `messages` est l'historique complet `[{role, content}]` (le dernier etant la
    question courante de l'utilisateur) ; seuls les derniers tours sont conserves
    pour rester dans une fenetre de contexte raisonnable. On termine par
    `Assistant:` pour amorcer la generation de la reponse.
    """
    recent = messages[-_HISTORY_TURNS:]
    lines = []
    for message in recent:
        role = "Utilisateur" if message.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {(message.get('content') or '').strip()}")
    lines.append("Assistant:")
    return CHAT_PROMPT_TEMPLATE.format(
        rules=_CITATION_RULES,
        context=_format_context(passages),
        history="\n".join(lines),
    )


_CITATION_RE = re.compile(r"\[(\d+)\]")


def cited_indices(answer: str, n_sources: int) -> list[int]:
    """Retourne les numeros de passages `[n]` reellement cites dans la reponse.

    Filtre sur `1..n_sources` (ignore les marqueurs hors borne, frequents avec un
    petit modele) et dedoublonne en conservant l'ordre d'apparition.
    """
    seen: list[int] = []
    for match in _CITATION_RE.findall(answer or ""):
        idx = int(match)
        if 1 <= idx <= n_sources and idx not in seen:
            seen.append(idx)
    return seen
