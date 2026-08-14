import pytest

from shorts_factory.daily_publish import DailyPublishLedger, DailyPublishLimitReached


def test_one_successful_publish_per_day(tmp_path):
    ledger = DailyPublishLedger(tmp_path / "ledger.json")
    first = ledger.reserve("soap")
    ledger.mark_published(first, "video-1")
    with pytest.raises(DailyPublishLimitReached):
        ledger.reserve("rope")


def test_failed_upload_allows_retry(tmp_path):
    ledger = DailyPublishLedger(tmp_path / "ledger.json")
    first = ledger.reserve("soap")
    ledger.mark_failed(first, "network error")
    assert ledger.reserve("soap")


def test_pending_upload_blocks_concurrent_publish(tmp_path):
    ledger = DailyPublishLedger(tmp_path / "ledger.json")
    ledger.reserve("soap")
    with pytest.raises(DailyPublishLimitReached):
        ledger.reserve("rope")
