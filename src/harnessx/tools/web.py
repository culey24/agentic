"""Web tools: fetch/browse a URL and (optional) search.

The search tool requires a provider-specific API key (e.g. Brave / Tavily).
It raises a clear error when no key is configured, so GAIA runs can degrade to
browse-only rather than silently failing.
"""

from __future__ import annotations

import os
import re

import httpx


async def browse(url: str, max_chars: int = 8000) -> str:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "harnessx/0.1"})
        resp.raise_for_status()
        html = resp.text
    text = _html_to_text(html)
    return text[:max_chars]


async def search(query: str, max_results: int = 5) -> str:
    key = os.environ.get("BRAVE_API_KEY") or os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("search requires BRAVE_API_KEY or TAVILY_API_KEY")
    if os.environ.get("TAVILY_API_KEY"):
        return await _tavily_search(query, key, max_results)
    return await _brave_search(query, key, max_results)


async def _tavily_search(query: str, key: str, max_results: int) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": key,
                "query": query,
                "max_results": max_results,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    results = data.get("results", [])
    lines = [f"{r.get('title', '')}: {r.get('content', '')}" for r in results]
    return "\n\n".join(lines)


async def _brave_search(query: str, key: str, max_results: int) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"X-Subscription-Token": key},
        )
        resp.raise_for_status()
        data = resp.json()
    results = data.get("web", {}).get("results", [])
    lines = [
        f"{r.get('title', '')}: {r.get('description', '')}" for r in results
    ]
    return "\n\n".join(lines)


def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = re.sub(r"&nbsp;", " ", html)
    html = re.sub(r"&amp;", "&", html)
    html = re.sub(r"&#\d+;", " ", html)
    return re.sub(r"\s+", " ", html).strip()
