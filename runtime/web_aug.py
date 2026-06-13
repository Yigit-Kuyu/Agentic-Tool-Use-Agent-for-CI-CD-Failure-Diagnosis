"""Test-time web augmentation for diagnosis results."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from html import unescape
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from .inference import DiagnosisRunResult, load_diagnosis_result, render_final_answer, run_diagnosis_from_path

try:
    import requests
except ImportError:  # pragma: no cover - requests is available here, but keep runtime optional.
    requests = None


TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
RESULT_RE = re.compile(
    r'<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>(?P<body>.*?)(?=<a[^>]*class="[^"]*result-link|\Z)',
    re.DOTALL,
)
HTML_RESULT_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>(?P<body>.*?)(?=<a[^>]*class="[^"]*result__a|\Z)',
    re.DOTALL,
)
SNIPPET_RE = re.compile(
    r'<td[^>]*class="[^"]*result-snippet[^"]*"[^>]*>(?P<snippet>.*?)</td>|<a[^>]*class="[^"]*result-snippet[^"]*"[^>]*>(?P<alt_snippet>.*?)</a>',
    re.DOTALL,
)
HTML_SNIPPET_RE = re.compile(
    r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>|<div[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(?P<alt_snippet>.*?)</div>',
    re.DOTALL,
)


class WebSearchClient(Protocol):
    """Small search interface so live search and tests share one contract."""

    provider_name: str

    def search(self, query: str, *, max_results: int = 5) -> list["WebSearchHit"]:
        ...


@dataclass(frozen=True)
class WebSearchHit:
    """One web result returned during test-time augmentation."""

    title: str
    url: str
    snippet: str
    provider: str
    rank: int


@dataclass(frozen=True)
class WebAugmentedAnswer:
    """Final answer enriched with references gathered after diagnosis."""

    failure_type: str
    diagnosis: str
    fix: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    web_findings: list[str] = field(default_factory=list)
    recommended_references: list[dict[str, str]] = field(default_factory=list)
    augmentation_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "diagnosis": self.diagnosis,
            "fix": self.fix,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "web_findings": list(self.web_findings),
            "recommended_references": list(self.recommended_references),
            "augmentation_note": self.augmentation_note,
        }


@dataclass(frozen=True)
class WebAugmentationResult:
    """Saved artifact for a diagnosis run plus optional live-search enrichment."""

    diagnosis_result: DiagnosisRunResult
    search_provider: str
    search_status: str
    search_error: str | None
    search_query_used: str | None
    search_queries_attempted: list[str]
    web_hits: list[WebSearchHit]
    enriched_final_answer: dict[str, Any]


class SearchUnavailableError(RuntimeError):
    """Raised when live search cannot be completed."""


class NoSearchResultsError(RuntimeError):
    """Raised when a live provider responded but returned no usable hits."""


class DuckDuckGoSearchClient:
    """Minimal live-search client using DuckDuckGo's lightweight HTML endpoint."""

    provider_name = "duckduckgo"

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchHit]:
        if requests is None:
            raise SearchUnavailableError("requests is not installed, so live web search is unavailable.")

        headers = {
            "User-Agent": "agi-tool/0.1 (+https://example.invalid/test-time-web-augmentation)",
        }
        errors: list[str] = []

        for endpoint_kind, url in (
            ("lite", f"https://lite.duckduckgo.com/lite/?q={quote(query)}"),
            ("html", f"https://html.duckduckgo.com/html/?q={quote(query)}"),
        ):
            try:
                response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
                response.raise_for_status()
            except Exception as exc:  # pragma: no cover - depends on live network conditions.
                errors.append(f"{endpoint_kind}: {exc}")
                continue

            hits = self._parse_results(response.text, max_results=max_results, endpoint_kind=endpoint_kind)
            if hits:
                return hits

        if errors:
            raise SearchUnavailableError(" | ".join(errors))
        raise NoSearchResultsError(f"DuckDuckGo returned no usable results for query: {query}")

    def _parse_results(self, html_text: str, *, max_results: int, endpoint_kind: str) -> list[WebSearchHit]:
        hits: list[WebSearchHit] = []
        result_re = RESULT_RE if endpoint_kind == "lite" else HTML_RESULT_RE
        snippet_re = SNIPPET_RE if endpoint_kind == "lite" else HTML_SNIPPET_RE
        for match in result_re.finditer(html_text):
            url = unescape(match.group("url")).strip()
            title = _clean_html(match.group("title"))
            body = match.group("body") or ""
            snippet_match = snippet_re.search(body)
            snippet = _clean_html(
                snippet_match.group("snippet") or snippet_match.group("alt_snippet") if snippet_match else ""
            )
            if not title or not url:
                continue
            hits.append(
                WebSearchHit(
                    title=title,
                    url=url,
                    snippet=snippet,
                    provider=self.provider_name,
                    rank=len(hits) + 1,
                )
            )
            if len(hits) >= max_results:
                break
        return hits


class BingSearchClient:
    """Minimal HTML search client using Bing's public search results page."""

    provider_name = "bing"

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchHit]:
        if requests is None:
            raise SearchUnavailableError("requests is not installed, so live web search is unavailable.")

        url = f"https://www.bing.com/search?q={quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; agi-tool/0.1; +https://example.invalid/test-time-web-augmentation)",
        }
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - depends on live network conditions.
            raise SearchUnavailableError(str(exc)) from exc

        hits = self._parse_results(response.text, max_results=max_results)
        if not hits:
            raise NoSearchResultsError(f"Bing returned no usable results for query: {query}")
        return hits

    def _parse_results(self, html_text: str, *, max_results: int) -> list[WebSearchHit]:
        result_re = re.compile(
            r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>.*?<h2><a href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a></h2>(?P<body>.*?)</li>',
            re.DOTALL,
        )
        snippet_re = re.compile(
            r'<p>(?P<snippet>.*?)</p>|<div[^>]*class="[^"]*b_caption[^"]*"[^>]*>.*?<p>(?P<alt_snippet>.*?)</p>',
            re.DOTALL,
        )

        hits: list[WebSearchHit] = []
        for match in result_re.finditer(html_text):
            url = unescape(match.group("url")).strip()
            title = _clean_html(match.group("title"))
            body = match.group("body") or ""
            snippet_match = snippet_re.search(body)
            snippet = _clean_html(
                snippet_match.group("snippet") or snippet_match.group("alt_snippet") if snippet_match else ""
            )
            if not title or not url:
                continue
            hits.append(
                WebSearchHit(
                    title=title,
                    url=url,
                    snippet=snippet,
                    provider=self.provider_name,
                    rank=len(hits) + 1,
                )
            )
            if len(hits) >= max_results:
                break
        return hits


class CascadingSearchClient:
    """Try multiple live providers in order until one returns usable hits."""

    provider_name = "cascade"

    def __init__(self, providers: list[WebSearchClient] | None = None) -> None:
        self.providers = providers or [DuckDuckGoSearchClient(), BingSearchClient()]

    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchHit]:
        provider_errors: list[str] = []
        for provider in self.providers:
            try:
                hits = provider.search(query, max_results=max_results)
                if hits:
                    return hits
            except NoSearchResultsError as exc:
                provider_errors.append(f"{provider.provider_name}: {exc}")
                continue
            except SearchUnavailableError as exc:
                provider_errors.append(f"{provider.provider_name}: {exc}")
                continue
        if provider_errors:
            raise SearchUnavailableError(" | ".join(provider_errors))
        raise NoSearchResultsError(f"No providers returned results for query: {query}")


class StaticSearchClient:
    """In-memory provider used for tests and offline demos."""

    provider_name = "static"

    def __init__(self, hits: list[WebSearchHit]) -> None:
        self._hits = list(hits)

    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchHit]:
        del query
        return list(self._hits[:max_results])


def augment_diagnosis_result(
    result: DiagnosisRunResult,
    *,
    search_client: WebSearchClient | None = None,
    max_results: int = 5,
) -> WebAugmentationResult:
    client = search_client or CascadingSearchClient()
    queries = _build_search_queries(result)

    if not queries:
        answer = _build_enriched_answer(result, web_hits=[], search_status="skipped")
        return WebAugmentationResult(
            diagnosis_result=result,
            search_provider=client.provider_name,
            search_status="skipped",
            search_error=None,
            search_query_used=None,
            search_queries_attempted=[],
            web_hits=[],
            enriched_final_answer=answer.to_dict(),
        )

    query_used: str | None = None
    try:
        web_hits: list[WebSearchHit] = []
        for query in queries:
            try:
                candidate_hits = client.search(query, max_results=max_results)
                query_used = query
                if candidate_hits:
                    web_hits = candidate_hits
                    break
            except NoSearchResultsError:
                query_used = query
                continue
        search_status = "ok" if web_hits else "no_results"
        search_error = None
    except SearchUnavailableError as exc:
        web_hits = []
        search_status = "unavailable"
        search_error = str(exc)
    except Exception as exc:
        # Web augmentation should never block the base diagnosis from being returned to the user.
        web_hits = []
        search_status = "unavailable"
        search_error = str(exc)

    answer = _build_enriched_answer(
        result,
        web_hits=web_hits,
        search_status=search_status,
        query_used=query_used,
    )
    provider_name = web_hits[0].provider if web_hits else client.provider_name
    return WebAugmentationResult(
        diagnosis_result=result,
        search_provider=provider_name,
        search_status=search_status,
        search_error=search_error,
        search_query_used=query_used,
        search_queries_attempted=queries,
        web_hits=web_hits,
        enriched_final_answer=answer.to_dict(),
    )


def augment_diagnosis_from_result_path(
    result_path: str | Path,
    *,
    search_client: WebSearchClient | None = None,
    max_results: int = 5,
) -> WebAugmentationResult:
    result = load_diagnosis_result(result_path)
    return augment_diagnosis_result(result, search_client=search_client, max_results=max_results)


def run_and_augment_diagnosis(
    *,
    policy_path: str | Path,
    dataset_root: str | Path | None = None,
    case_id: str | None = None,
    split: str = "test",
    max_steps: int = 5,
    seed: int | None = None,
    search_client: WebSearchClient | None = None,
    max_results: int = 5,
) -> WebAugmentationResult:
    result = run_diagnosis_from_path(
        policy_path=policy_path,
        dataset_root=dataset_root,
        case_id=case_id,
        split=split,
        max_steps=max_steps,
        seed=seed,
    )
    return augment_diagnosis_result(result, search_client=search_client, max_results=max_results)


def save_web_augmentation_result(result: WebAugmentationResult, output_path: str | Path) -> Path:
    output = Path(output_path)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)
    return output


def render_web_augmented_answer(result: WebAugmentationResult) -> str:
    base = result.diagnosis_result
    answer = result.enriched_final_answer
    findings = answer.get("web_findings", [])
    findings_text = "\n".join(f"- {item}" for item in findings) if findings else "- None"
    references = answer.get("recommended_references", [])
    references_text = (
        "\n".join(f"- {item.get('title', '')}: {item.get('url', '')}" for item in references)
        if references
        else "- None"
    )
    return (
        f"{render_final_answer(base)}\n"
        f"Web augmentation status: {result.search_status}\n"
        f"Web query used: {result.search_query_used or 'None'}\n"
        f"Web findings:\n{findings_text}\n"
        f"Recommended references:\n{references_text}\n"
        f"Augmentation note: {answer.get('augmentation_note', '')}"
    )


def _build_enriched_answer(
    result: DiagnosisRunResult,
    *,
    web_hits: list[WebSearchHit],
    search_status: str,
    query_used: str | None,
) -> WebAugmentedAnswer:
    answer = dict(result.final_answer)
    base_evidence = list(answer.get("evidence", []))
    confidence = float(answer.get("confidence", 0.0))

    if not web_hits:
        if search_status == "no_results":
            fallback_refs = _fallback_references(result)
            note = "No live web references were found, so category-level fallback references were added."
            return WebAugmentedAnswer(
                failure_type=answer.get("failure_type", ""),
                diagnosis=answer.get("diagnosis", ""),
                fix=answer.get("fix", ""),
                evidence=base_evidence,
                confidence=confidence,
                web_findings=[],
                recommended_references=fallback_refs,
                augmentation_note=note,
            )
        elif search_status == "skipped":
            note = "This case did not define a web augmentation query, so the internal diagnosis is returned unchanged."
        else:
            fallback_refs = _fallback_references(result)
            note = (
                "Live web augmentation was unavailable, so category-level fallback references were added "
                "instead of live search results."
            )
            return WebAugmentedAnswer(
                failure_type=answer.get("failure_type", ""),
                diagnosis=answer.get("diagnosis", ""),
                fix=answer.get("fix", ""),
                evidence=base_evidence,
                confidence=confidence,
                web_findings=[],
                recommended_references=fallback_refs,
                augmentation_note=note,
            )
        return WebAugmentedAnswer(
            failure_type=answer.get("failure_type", ""),
            diagnosis=answer.get("diagnosis", ""),
            fix=answer.get("fix", ""),
            evidence=base_evidence,
            confidence=confidence,
            web_findings=[],
            recommended_references=[],
            augmentation_note=note,
        )

    # Keep augmentation additive: the internal diagnosis stays primary, and the web layer only
    # contributes supporting references so train-time behavior remains cleanly separated.
    findings = [_format_web_finding(hit) for hit in web_hits]
    references = [{"title": hit.title, "url": hit.url} for hit in web_hits]
    note = (
        f"Added {len(web_hits)} web references for post-diagnosis validation and implementation guidance"
        + (f" using query: {query_used}." if query_used else ".")
    )
    return WebAugmentedAnswer(
        failure_type=answer.get("failure_type", ""),
        diagnosis=answer.get("diagnosis", ""),
        fix=answer.get("fix", ""),
        evidence=base_evidence,
        confidence=min(1.0, confidence + 0.05),
        web_findings=findings,
        recommended_references=references,
        augmentation_note=note,
    )


def _format_web_finding(hit: WebSearchHit) -> str:
    if hit.snippet:
        return f"{hit.title}: {hit.snippet}"
    return hit.title


def _clean_html(value: str) -> str:
    stripped = TAG_RE.sub(" ", value)
    return WHITESPACE_RE.sub(" ", unescape(stripped)).strip()


def _build_search_queries(result: DiagnosisRunResult) -> list[str]:
    queries: list[str] = []

    def add(query: str) -> None:
        normalized = WHITESPACE_RE.sub(" ", query).strip()
        if normalized and normalized not in queries:
            queries.append(normalized)

    if result.web_augmentation_query.strip():
        add(result.web_augmentation_query)

    answer = result.final_answer
    diagnosis = str(answer.get("diagnosis", "")).strip()
    fix = str(answer.get("fix", "")).strip()
    evidence = list(answer.get("evidence", []))
    top_evidence = str(evidence[0]).strip() if evidence else ""
    category = result.category.replace("_", " ")

    if top_evidence:
        add(f"{top_evidence} GitHub Actions fix")
        add(f"{top_evidence} Python CI fix")
    if diagnosis:
        add(f"{diagnosis} GitHub Actions")
    if fix:
        add(f"{fix} GitHub Actions")
    add(f"{category} GitHub Actions troubleshooting")
    add(f"{category} CI/CD fix")
    return queries[:6]


def _fallback_references(result: DiagnosisRunResult) -> list[dict[str, str]]:
    category = result.category
    defaults = {
        "missing_dependency": [
            {"title": "GitHub Actions Python packaging guide", "url": "https://docs.github.com/actions/automating-builds-and-tests/building-and-testing-python"},
            {"title": "pip install requirements file usage", "url": "https://pip.pypa.io/en/stable/reference/requirements-file-format/"},
        ],
        "wrong_python_version": [
            {"title": "actions/setup-python documentation", "url": "https://github.com/actions/setup-python"},
            {"title": "GitHub Actions Python workflow guide", "url": "https://docs.github.com/actions/automating-builds-and-tests/building-and-testing-python"},
        ],
        "import_path_error": [
            {"title": "pytest good practices for import paths", "url": "https://docs.pytest.org/en/stable/explanation/goodpractices.html"},
            {"title": "Python module search path documentation", "url": "https://docs.python.org/3/tutorial/modules.html"},
        ],
        "file_not_found": [
            {"title": "GitHub Actions workflow syntax", "url": "https://docs.github.com/actions/using-workflows/workflow-syntax-for-github-actions"},
            {"title": "Dockerfile COPY reference", "url": "https://docs.docker.com/reference/dockerfile/"},
        ],
        "docker_build_failure": [
            {"title": "Dockerfile reference", "url": "https://docs.docker.com/reference/dockerfile/"},
            {"title": "Docker build concepts", "url": "https://docs.docker.com/build/concepts/dockerfile/"},
        ],
        "failed_unit_test": [
            {"title": "pytest documentation", "url": "https://docs.pytest.org/en/stable/"},
            {"title": "GitHub Actions Python testing guide", "url": "https://docs.github.com/actions/automating-builds-and-tests/building-and-testing-python"},
        ],
        "missing_env_variable": [
            {"title": "GitHub Actions variables", "url": "https://docs.github.com/actions/learn-github-actions/variables"},
            {"title": "GitHub Actions secrets", "url": "https://docs.github.com/actions/security-guides/encrypted-secrets"},
        ],
        "bad_github_actions_yaml": [
            {"title": "GitHub Actions workflow syntax", "url": "https://docs.github.com/actions/using-workflows/workflow-syntax-for-github-actions"},
            {"title": "GitHub Actions contexts and expressions", "url": "https://docs.github.com/actions/learn-github-actions/contexts"},
        ],
    }
    return defaults.get(category, [])
