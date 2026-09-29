"""Chooses the chat model from environment variables.

EXTRACTION_MODEL_PROVIDER = offline (default) | openai | azure
  openai: OPENAI_API_KEY, optional OPENAI_MODEL (default gpt-4o-mini)
  azure:  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, optional OPENAI_API_VERSION
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from doc_extraction.offline_model import OfflineExtractionModel


def get_model(provider: str | None = None) -> BaseChatModel:
    provider = (provider or os.getenv("EXTRACTION_MODEL_PROVIDER", "offline")).lower()

    if provider == "offline":
        return OfflineExtractionModel()

    try:
        from langchain_openai import AzureChatOpenAI, ChatOpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install the OpenAI extra: pip install -e '.[openai]'") from exc

    if provider == "openai":
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    if provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version=os.getenv("OPENAI_API_VERSION", "2024-10-21"),
            temperature=0,
        )
    raise ValueError(f"Unknown EXTRACTION_MODEL_PROVIDER '{provider}'")
