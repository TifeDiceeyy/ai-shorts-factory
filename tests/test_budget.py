import pytest
from shorts_factory.cost_tracker import BudgetExceeded, CostTracker


class FakePaidProvider:
    """Simulates a real (non-stub) provider with a nonzero per-call cost, to
    prove the budget guard fires BEFORE the request — not just that a report
    ends up over cap after the fact."""

    def __init__(self, cost_per_call: float):
        self.cost_per_call = cost_per_call
        self.call_count = 0

    def call(self, tracker: CostTracker, operation: str = "fake.paid_call"):
        tracker.check_budget(operation, self.cost_per_call)  # may raise, BEFORE the "request"
        self.call_count += 1  # stands in for "the paid HTTP request happened"
        tracker.record("fake", operation, self.cost_per_call, self.cost_per_call, is_stub=False)


def test_call_within_budget_succeeds():
    tracker = CostTracker(budget_cap_usd=1.00)
    provider = FakePaidProvider(cost_per_call=0.30)
    provider.call(tracker)
    provider.call(tracker)
    provider.call(tracker)
    assert provider.call_count == 3
    assert tracker.total_spent_usd == pytest.approx(0.90)


def test_call_that_would_exceed_budget_aborts_before_request():
    tracker = CostTracker(budget_cap_usd=1.00)
    provider = FakePaidProvider(cost_per_call=0.60)

    provider.call(tracker)  # $0.60 spent, under cap
    assert provider.call_count == 1

    with pytest.raises(BudgetExceeded):
        provider.call(tracker)  # would bring total to $1.20 > $1.00 cap

    # The critical assertion: call_count did NOT increment. The guard fired
    # before the (simulated) request, so no paid call was made.
    assert provider.call_count == 1
    assert tracker.total_spent_usd == pytest.approx(0.60)


def test_zero_cap_blocks_any_nonzero_paid_call():
    tracker = CostTracker(budget_cap_usd=0.0)
    provider = FakePaidProvider(cost_per_call=0.01)
    with pytest.raises(BudgetExceeded):
        provider.call(tracker)
    assert provider.call_count == 0


def test_stub_providers_never_trip_the_guard():
    """Stub providers report $0 cost, so even a tiny cap should never block
    them — this is what lets the whole Phase 0 pipeline run for free."""
    tracker = CostTracker(budget_cap_usd=0.0)
    tracker.check_budget("stub.op", estimated_cost_usd=0.0)  # must not raise
