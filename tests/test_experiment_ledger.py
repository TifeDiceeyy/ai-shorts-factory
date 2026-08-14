"""Phase 6: prove the scoring formula reproduces by hand (CLAUDE.md gate:
"scores reproduce by hand for one video"), and that the ledger records/
ingests correctly."""
from shorts_factory.experiment_ledger import compute_score, ingest_metrics, load_ledger, record_publish
from shorts_factory.providers.youtube_analytics import VideoMetrics


def _metrics(views, avg_view_pct, subs_gained):
    return VideoMetrics(
        video_id="test-vid",
        start_date="2026-08-01",
        end_date="2026-08-14",
        views=views,
        estimated_minutes_watched=0.0,
        average_view_duration_seconds=0.0,
        average_view_percentage=avg_view_pct,
        subscribers_gained=subs_gained,
        subscribers_lost=0,
        raw_response={},
    )


def test_score_reproduces_by_hand_for_one_video():
    # 1000 views, 65% average retention, 8 new subscribers.
    metrics = _metrics(views=1000, avg_view_pct=65.0, subs_gained=8)

    # By hand: retention_component = 65/100 = 0.65
    #          sub_rate = 8/1000 = 0.008 -> sub_component = min(1, 0.008*100) = 0.8
    #          score = 0.7*0.65 + 0.3*0.8 = 0.455 + 0.24 = 0.695
    expected = round(0.7 * 0.65 + 0.3 * 0.8, 4)

    assert compute_score(metrics) == expected == 0.695


def test_score_caps_sub_component_at_one():
    # An improbably high sub rate (5%) must still cap the sub component at 1.0.
    metrics = _metrics(views=100, avg_view_pct=50.0, subs_gained=5)
    # sub_rate = 0.05 -> *100 = 5 -> capped to 1.0
    expected = round(0.7 * 0.5 + 0.3 * 1.0, 4)
    assert compute_score(metrics) == expected


def test_score_handles_zero_views_without_division_error():
    metrics = _metrics(views=0, avg_view_pct=0.0, subs_gained=0)
    assert compute_score(metrics) == 0.0


def test_record_then_ingest_updates_the_same_entry(tmp_path, monkeypatch):
    ledger_path = tmp_path / "experiment_ledger.json"
    monkeypatch.setattr("shorts_factory.experiment_ledger.LEDGER_PATH", ledger_path)

    record_publish("soap", "vid123", concept="How to reinvent soap", hook_variant_index=2, series="reinvent-it")
    assert len(load_ledger()) == 1

    metrics = _metrics(views=500, avg_view_pct=70.0, subs_gained=3)
    entry = ingest_metrics("vid123", metrics)

    assert entry["metrics"]["views"] == 500
    assert entry["score"] == compute_score(metrics)
    assert len(load_ledger()) == 1  # updated in place, not duplicated
