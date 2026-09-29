from __future__ import annotations

from pathlib import Path

import pytest

from doc_extraction.offline_model import OfflineExtractionModel
from doc_extraction.pipeline import ExtractionPipeline

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def pipeline() -> ExtractionPipeline:
    return ExtractionPipeline(OfflineExtractionModel())


@pytest.fixture
def sample():
    def read(name: str) -> str:
        return (DATA / "samples" / name).read_text(encoding="utf-8")

    return read
