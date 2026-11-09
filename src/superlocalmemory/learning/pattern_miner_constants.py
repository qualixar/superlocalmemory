# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — F4.A Stage-8 H-01 fix

"""Static dictionaries used by ``pattern_miner`` — extracted so the
main module stays under the 400-LOC cap.
"""

from __future__ import annotations

TECH_KEYWORDS: dict[str, str] = {
    "python": "Python", "javascript": "JavaScript",
    "typescript": "TypeScript", "react": "React",
    "vue": "Vue", "angular": "Angular",
    "postgresql": "PostgreSQL", "mysql": "MySQL",
    "sqlite": "SQLite", "docker": "Docker",
    "kubernetes": "Kubernetes", "aws": "AWS",
    "azure": "Azure", "gcp": "GCP",
    "node": "Node.js", "fastapi": "FastAPI",
    "django": "Django", "flask": "Flask",
    "rust": "Rust", "go": "Go", "java": "Java",
    "git": "Git", "npm": "npm", "pip": "pip",
    "langchain": "LangChain", "ollama": "Ollama",
    "pytorch": "PyTorch", "claude": "Claude",
    "openai": "OpenAI", "anthropic": "Anthropic",
    "redis": "Redis", "mongodb": "MongoDB",
    "graphql": "GraphQL", "nextjs": "Next.js",
    "terraform": "Terraform", "nginx": "Nginx",
    "linux": "Linux", "macos": "macOS",
    "vscode": "VS Code", "neovim": "Neovim",
    # The spellings people actually type. Whole-word matching fixed a real
    # defect — "going" no longer counts as Go — and cost these, because there
    # is no word boundary inside "golang" or "nodejs". They are listed rather
    # than matched by prefix, because a prefix rule brings the original problem
    # straight back.
    "golang": "Go", "nodejs": "Node.js", "node.js": "Node.js",
    "reactjs": "React", "react.js": "React",
    "vuejs": "Vue", "vue.js": "Vue",
    "next.js": "Next.js", "nuxtjs": "Nuxt", "nuxt": "Nuxt",
    "postgres": "PostgreSQL", "k8s": "Kubernetes",
    "typescript": "TypeScript", "ts": "TypeScript",
    "golang.org": "Go",
}


STOPWORDS: frozenset[str] = frozenset({
    "the", "is", "a", "an", "in", "on", "at", "to", "for",
    "of", "and", "or", "not", "with", "that", "this", "was",
    "are", "be", "has", "had", "have", "from", "by", "it",
    "its", "as", "but", "were", "been", "being", "would",
    "could", "should", "will", "may", "might", "can", "do",
    "does", "did", "about", "into", "over", "after", "before",
    "then", "than", "also", "just", "like", "more", "some",
    "only", "other", "such", "each", "every", "both", "most",
    # Pronouns and subordinators. Their absence is why "their" and "while"
    # became recorded interests on a live store, at confidence 1.0, and were
    # then rendered into a prompt injected on every turn. A word that appears
    # in most English sentences tells you nothing about the person writing them.
    "their", "them", "they", "these", "those", "there", "while", "when",
    "where", "which", "who", "whom", "whose", "what", "why", "how",
    "he", "she", "him", "her", "his", "hers", "we", "us", "our", "ours",
    "you", "your", "yours", "i", "me", "my", "mine", "myself",
    "if", "else", "because", "since", "until", "unless", "though",
    "although", "however", "therefore", "thus", "here", "very", "much",
    "many", "same", "own", "too", "any", "all", "none", "nor", "yet",
    "so", "up", "down", "out", "off", "again", "once", "still",
})


def _augment_with_shared_list() -> frozenset[str]:
    """Fold in the larger stopword list this codebase already maintains.

    ``core.topic_signature`` carries a longer list, and it contained both of the
    words that leaked through here. Two lists of the same thing is how one ends
    up worse than the other, so this reads that one rather than restating it —
    and keeps working if it ever moves, because a missing import degrades to the
    list above instead of failing at import time.
    """
    try:
        from superlocalmemory.core.topic_signature import _STOPWORDS as _shared
    except Exception:  # pragma: no cover — the local list still applies
        return STOPWORDS
    return STOPWORDS | frozenset(_shared)


STOPWORDS = _augment_with_shared_list()


__all__ = ("TECH_KEYWORDS", "STOPWORDS")
