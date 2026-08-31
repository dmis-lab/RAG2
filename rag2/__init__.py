"""Reproduction of the original RAG2 system.

RAG2: Rationale-Guided Retrieval Augmented Generation for Medical Question
Answering (Sohn et al., NAACL 2025). See docs/rag2_reproduction.md for what the
paper specifies, what it leaves open, and every assumption this code makes.

This package is the reproduction. The authors' released code is kept unmodified
under ``retriever/`` and ``classifier/``.
"""

__version__ = "0.1.0"

from .config import Config, load_config  # noqa: F401
from .schema import CandidateSet, Evidence, FilterDecision, PipelineResult, Question  # noqa: F401
