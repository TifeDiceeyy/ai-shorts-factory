"""Web search provider interface for Phase 1 retrieval.

Unlike LLM/TTS/image, there is no meaningful local "stub" here: a fake
search result can't produce a real, checkable citation, and citations being
real and checkable is the entire point of Phase 1. So this module has
exactly one runtime-supported provider (Tavily) and no zero-cost fallback —
if no real SEARCH_PROVIDER + SEARCH_API_KEY is configured, retrieval simply
refuses to run (see get_search_provider()). Tests use a separate, explicitly
test-only FakeSearchProvider (see tests/) with canned results, so the suite
doesn't depend on network access or a live key.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

from ..cost_tracker import CostTracker

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
# Real pricing as of Aug 2026 (pay-as-you-go): basic search = 1 credit =
# $0.008. Advanced search = 2 credits = $0.016. We use basic search.
TAVILY_BASIC_SEARCH_COST_USD = 0.008


@dataclass
class SearchResult:
    url: str
    title: str
    content: str
    domain: str
    score: float | None = None


class SearchProvider(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, max_results: int, cost_tracker: CostTracker) -> list[SearchResult]:
        ...


def _domain_of(url: str) -> str:
    from urllib.parse import urlparse
    netloc = urlparse(url).netloc
    return netloc.removeprefix("www.")


class TavilySearchProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("TavilySearchProvider requires a non-empty api_key")
        self.api_key = api_key

    def search(self, query: str, max_results: int, cost_tracker: CostTracker) -> list[SearchResult]:
        operation = f"search.tavily[{query!r}]"
        cost_tracker.check_budget(operation, TAVILY_BASIC_SEARCH_COST_USD)

        resp = requests.post(
            TAVILY_SEARCH_URL,
            json={
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_raw_content": True,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("results", []):
            url = item.get("url", "")
            text = item.get("raw_content") or item.get("content") or ""
            results.append(
                SearchResult(
                    url=url,
                    title=item.get("title", ""),
                    content=text,
                    domain=_domain_of(url),
                    score=item.get("score"),
                )
            )

        cost_tracker.record(
            provider=self.name,
            operation=operation,
            estimated_cost_usd=TAVILY_BASIC_SEARCH_COST_USD,
            actual_cost_usd=TAVILY_BASIC_SEARCH_COST_USD,
            is_stub=False,
        )
        return results


def get_search_provider(provider_name: str, api_key: str) -> SearchProvider:
    name = provider_name.strip().lower()
    if name == "tavily":
        return TavilySearchProvider(api_key)
    raise NotImplementedError(
        f"Search provider {provider_name!r} is not implemented. Only 'tavily' is "
        "currently supported. There is no stub option — real retrieval needs a "
        "real, keyed search provider to produce checkable citations."
    )
