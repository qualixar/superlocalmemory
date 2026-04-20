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
})


__all__ = ("TECH_KEYWORDS", "STOPWORDS")
