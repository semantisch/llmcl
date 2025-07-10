"""
LLMCL - LLM Clue Extractor
A tool for extracting ASP clues from natural language puzzle descriptions.
"""

from .llm_connector import LLMConnector
from .asp_generator import ASPGenerator

__version__ = "1.0.0"
__author__ = "BilAI Summer School 2025"

__all__ = ["LLMConnector", "ASPGenerator"] 