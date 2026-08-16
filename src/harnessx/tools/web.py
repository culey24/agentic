"""Web tools: fetch/browse a URL and search.

Search prefers Brave / Tavily when a key is configured, otherwise falls back to
DuckDuckGo HTML (no key required) so GAIA-style agents can run without extra
credentials.
"""

from __future__ import annotations

import html as _html
import os
import re

import httpx


async def browse(url: str, max_chars: int = 8000) -> str:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "harnessx/0.1"})
        resp.raise_for_status()
        content = resp.text
    text = _html_to_text(content)
    return text[:max_chars]


async def search(query: str, max_results: int = 5) -> str:
    if os.environ.get("TAVILY_API_KEY"):
        return await _tavily_search(query, os.environ["TAVILY_API_KEY"], max_results)
    if os.environ.get("BRAVE_API_KEY"):
        return await _brave_search(query, os.environ["BRAVE_API_KEY"], max_results)
    return await _ddg_search(query, max_results)


async def _ddg_search(query: str, max_results: int) -> str:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
        )
        resp.raise_for_status()
        content = resp.text
    results = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        content,
        re.DOTALL,
    )
    snippets = re.findall(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', content, re.DOTALL
    )
    lines = []
    for i, (url, title) in enumerate(results[:max_results]):
        snippet = (
            _html.unescape(re.sub(r"<[^>]+>", "", snippets[i])).strip()
            if i < len(snippets)
            else ""
        )
        title = _html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
        lines.append(f"{title}: {url} - {snippet}")
    if not lines:
        return "No results found."
    return "\n\n".join(lines)


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
