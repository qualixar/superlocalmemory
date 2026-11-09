# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.3

"""Phase F: The Learning Brain — Memory Parameterization.

Converts consolidated knowledge into soft prompts (pure text personality encoding).
No LoRA, no model weights, no cloud. Local-first LTM-Implicit procedural memory.

Components:
  - PatternExtractor: Mines patterns from core memory, behavioral, cross-project, workflows
  - SoftPromptGenerator: Converts patterns to natural language soft prompt templates
  - PromptInjector: Injects soft prompts into context with token budget management
  - PromptLifecycleManager: Ebbinghaus decay + effectiveness tracking for prompts
  - PIIFilter: Stateless PII detection and redaction

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from superlocalmemory.parameterization.pattern_extractor import (
    PatternAssertion,
    PatternCategory,
    PatternExtractor,
)
from superlocalmemory.parameterization.pii_filter import PII_PATTERNS, PIIFilter
from superlocalmemory.parameterization.prompt_injector import PromptInjector
from superlocalmemory.parameterization.prompt_lifecycle import PromptLifecycleManager
from superlocalmemory.parameterization.soft_prompt_generator import (
    CATEGORY_PRIORITY_ORDER,
    CATEGORY_TEMPLATES,
    SoftPromptGenerator,
    SoftPromptTemplate,
)

__all__ = [
    "PatternAssertion",
    "PatternCategory",
    "PatternExtractor",
    "SoftPromptGenerator",
    "SoftPromptTemplate",
    "CATEGORY_TEMPLATES",
    "CATEGORY_PRIORITY_ORDER",
    "PromptInjector",
    "PromptLifecycleManager",
    "PIIFilter",
    "PII_PATTERNS",
]
