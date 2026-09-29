"""Command line.

doc-extract run data/samples/*.txt     extract and show the decision for each file
doc-extract eval                       evaluate against data/labels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc_extraction.evaluate import evaluate, report
from doc_extraction.loaders import load
from doc_extraction.models import get_model
from doc_extraction.pipeline import ExtractionPipeline


def _data_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "labels.json").is_file():
            return parent / "data"
    return Path("data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured data from invoices and contracts.")
    parser.add_argument("--provider", help="offline | openai | azure (default: EXTRACTION_MODEL_PROVIDER or offline)")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="extract data from files")
    run.add_argument("files", nargs="+", type=Path)
    run.add_argument("--json", action="store_true", help="print the full JSON result for each file")
    sub.add_parser("eval", help="evaluate against the labelled samples")
    args = parser.parse_args()

    pipeline = ExtractionPipeline(get_model(args.provider))

    if args.command == "eval":
        data = _data_dir()
        print(report(evaluate(pipeline, data / "labels.json", data / "samples")))
        return

    for path in args.files:
        result = pipeline.run(load(path), path.name)
        if args.json:
            print(json.dumps(result.model_dump(mode="json"), indent=2))
            continue
        print(f"{path.name:16} {result.document_type.value:9} {result.status.value}")
        for reason in result.review_reasons:
            print(f"{'':16} - {reason}")


if __name__ == "__main__":
    main()
