from conftest import DATA

from doc_extraction.evaluate import evaluate, field_matches


def test_field_comparison_rules():
    assert field_matches("total", "2814.35", "2814.350")
    assert field_matches("vendor_name", "Acme GmbH", "ACME  GmbH")
    assert field_matches("governing_law", ["England", "English law"], "english law")
    assert field_matches("parties", ["A GmbH", "B BV"], [{"name": "B BV"}, {"name": "A GmbH"}])
    assert field_matches("due_date", None, None)
    assert not field_matches("due_date", None, "2025-01-01")


def test_offline_baseline_never_auto_approves_wrong_data(pipeline):
    scores = evaluate(pipeline, DATA / "labels.json", DATA / "samples")

    assert not [s.source for s in scores if s.unsafe_auto_approval]
    assert all(s.routed_ok for s in scores if s.split == "tuned")
