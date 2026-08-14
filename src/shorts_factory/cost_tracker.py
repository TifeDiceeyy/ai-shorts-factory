"""Cost ledger + budget guard.

Every provider operation — stub or real — is logged here. The guard check
happens BEFORE the operation runs, not after: `BudgetGuard.check()` raises
BudgetExceeded and the caller must not proceed to the actual provider call.
This is what makes "abort before an over-budget paid request" provable
rather than assumed: tests assert the provider's call count stays at zero
when the guard fires.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


class BudgetExceeded(Exception):
    def __init__(self, projected: float, cap: float, operation: str):
        self.projected = projected
        self.cap = cap
        self.operation = operation
        super().__init__(
            f"Aborting {operation!r}: projected total cost ${projected:.4f} "
            f"would exceed the ${cap:.4f} budget cap. No request was made."
        )


@dataclass
class CostEntry:
    provider: str
    operation: str
    estimated_cost_usd: float
    actual_cost_usd: float
    is_stub: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "operation": self.operation,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "actual_cost_usd": round(self.actual_cost_usd, 6),
            "is_stub": self.is_stub,
            "timestamp": self.timestamp,
        }


class CostTracker:
    def __init__(self, budget_cap_usd: float):
        self.budget_cap_usd = budget_cap_usd
        self.entries: list[CostEntry] = []

    @property
    def total_spent_usd(self) -> float:
        return sum(e.actual_cost_usd for e in self.entries)

    def check_budget(self, operation: str, estimated_cost_usd: float) -> None:
        """Call BEFORE making a provider call. Raises BudgetExceeded (and logs
        nothing, spends nothing) if the projected total would exceed the cap."""
        projected = self.total_spent_usd + estimated_cost_usd
        if projected > self.budget_cap_usd:
            raise BudgetExceeded(projected, self.budget_cap_usd, operation)

    def record(
        self,
        provider: str,
        operation: str,
        estimated_cost_usd: float,
        actual_cost_usd: float,
        is_stub: bool,
    ) -> None:
        self.entries.append(
            CostEntry(
                provider=provider,
                operation=operation,
                estimated_cost_usd=estimated_cost_usd,
                actual_cost_usd=actual_cost_usd,
                is_stub=is_stub,
            )
        )

    def write_report(self, path: Path) -> dict:
        report = {
            "budget_cap_usd": self.budget_cap_usd,
            "total_spent_usd": round(self.total_spent_usd, 6),
            "under_cap": self.total_spent_usd <= self.budget_cap_usd,
            "entries": [e.to_dict() for e in self.entries],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report
