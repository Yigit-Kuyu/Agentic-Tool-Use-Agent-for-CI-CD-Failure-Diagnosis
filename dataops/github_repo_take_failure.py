#!/usr/bin/env python3

import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


OWNER = ""
REPO = ""

OUT_DIR = Path("agi_tool/failed_workflow_logs")
RUNS_JSON = Path("agi_tool/runs.json")
SUMMARY_MD = Path("agi_tool/github_failure_summary.md")
SUMMARY_TSV = Path("agi_tool/github_failure_summary.tsv")
DOWNLOAD_ERRORS_TSV = Path("agi_tool/github_download_errors.tsv")


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[^\s]+\s+")

IGNORE_PATTERNS = [
    re.compile(r"warnings summary", re.I),
    re.compile(r"UserWarning", re.I),
    re.compile(r"Failed to initialize NumPy", re.I),
    re.compile(r"1 warning", re.I),
]


REASON_PATTERNS = [
    (500, "missing_required_env", re.compile(r"Missing required environment variable:\s*([A-Za-z_][A-Za-z0-9_]*)", re.I)),
    (450, "runtime_error", re.compile(r"RuntimeError:\s*(.*)", re.I)),
    (420, "docker_failed_build", re.compile(r"ERROR:\s*failed to build:.*", re.I)),
    (410, "docker_failed_solve", re.compile(r"failed to solve:.*", re.I)),
    (400, "docker_process_failed", re.compile(r"process .* did not complete successfully.*", re.I)),
    (350, "pytest_failed_summary", re.compile(r"FAILED tests?/.*", re.I)),
    (330, "python_exception", re.compile(r"(AssertionError|ModuleNotFoundError|ImportError|ValueError|TypeError|RuntimeError|FileNotFoundError):.*", re.I)),
    (250, "github_error", re.compile(r"##\[error\].*", re.I)),
    (100, "exit_code", re.compile(r"Process completed with exit code \d+|exit code:\s*\d+", re.I)),
]


def clean_line(line: str) -> str:
    line = ANSI_RE.sub("", line)
    line = TIMESTAMP_RE.sub("", line)
    return line.strip()


def safe_name(text: str) -> str:
    text = text or "run"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def github_request(url: str, token: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "github-actions-log-collector",
        },
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def fetch_all_runs(token: str) -> dict:
    all_runs = []
    page = 1
    total_count = None

    while True:
        url = (
            f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs"
            f"?per_page=100&page={page}"
        )

        print(f"Fetching workflow runs page {page}...")
        data = json.loads(github_request(url, token).decode("utf-8"))

        if total_count is None:
            total_count = data.get("total_count", 0)

        runs = data.get("workflow_runs", [])
        if not runs:
            break

        all_runs.extend(runs)

        if len(runs) < 100:
            break

        page += 1

    result = {
        "total_count": total_count,
        "workflow_runs": all_runs,
    }

    RUNS_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def download_failed_logs(runs_data: dict, token: str) -> list[dict]:
    OUT_DIR.mkdir(exist_ok=True)

    failed_runs = [
        r for r in runs_data.get("workflow_runs", [])
        if r.get("conclusion") == "failure"
    ]

    print(f"Failed runs found: {len(failed_runs)}")

    download_errors = []

    for idx, run in enumerate(failed_runs, start=1):
        run_id = str(run["id"])
        title = safe_name(
            run.get("display_title")
            or run.get("head_commit", {}).get("message")
            or run.get("name")
            or "run"
        )
        short_sha = (run.get("head_sha") or "unknown")[:7]
        created_at = safe_name(run.get("created_at", "unknown"))
        logs_url = run.get("logs_url")

        zip_path = OUT_DIR / f"{created_at}_{title}_{short_sha}_{run_id}.zip"

        if zip_path.exists() and zip_path.stat().st_size > 0:
            print(f"[{idx}/{len(failed_runs)}] Already exists: {zip_path.name}")
            continue

        print(f"[{idx}/{len(failed_runs)}] Downloading logs for run {run_id}...")

        try:
            content = github_request(logs_url, token)
            zip_path.write_bytes(content)

            if zip_path.stat().st_size == 0:
                raise RuntimeError("Downloaded ZIP is empty")

        except Exception as exc:
            print(f"  ERROR: could not download run {run_id}: {exc}")
            download_errors.append(
                {
                    "run_id": run_id,
                    "run_number": run.get("run_number", ""),
                    "title": title,
                    "url": run.get("html_url", ""),
                    "error": str(exc),
                }
            )

            if zip_path.exists() and zip_path.stat().st_size == 0:
                zip_path.unlink()

        time.sleep(0.2)

    if download_errors:
        with DOWNLOAD_ERRORS_TSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["run_number", "run_id", "title", "url", "error"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(download_errors)

        print(f"Download errors written to {DOWNLOAD_ERRORS_TSV}")

    return download_errors


def normalize_reason(reason_type: str, raw_line: str) -> str:
    clean = clean_line(raw_line)

    env_match = re.search(
        r"Missing required environment variable:\s*([A-Za-z_][A-Za-z0-9_]*)",
        clean,
        re.I,
    )
    if env_match:
        return f"Missing required environment variable: {env_match.group(1)}"

    runtime_match = re.search(r"RuntimeError:\s*(.*)", clean, re.I)
    if runtime_match:
        return f"RuntimeError: {runtime_match.group(1)}"

    return clean


def find_best_reason(text: str) -> tuple[str, str, str]:
    text = ANSI_RE.sub("", text)
    lines = text.splitlines()

    best = None

    for i, original_line in enumerate(lines):
        line = clean_line(original_line)

        if not line:
            continue

        if any(p.search(line) for p in IGNORE_PATTERNS):
            continue

        for score, reason_type, pattern in REASON_PATTERNS:
            if pattern.search(line):
                reason = normalize_reason(reason_type, line)

                start = max(0, i - 8)
                end = min(len(lines), i + 15)
                context = "\n".join(lines[start:end])

                candidate = {
                    "score": score,
                    "index": i,
                    "reason_type": reason_type,
                    "reason": reason,
                    "context": context,
                }

                if best is None:
                    best = candidate
                elif candidate["score"] > best["score"]:
                    best = candidate
                elif candidate["score"] == best["score"] and candidate["index"] > best["index"]:
                    best = candidate

    if best is None:
        return "unknown", "No clear failure reason found", ""

    return best["reason_type"], best["reason"], best["context"]


def summarize_logs(runs_data: dict) -> list[dict]:
    run_map = {
        str(r["id"]): r
        for r in runs_data.get("workflow_runs", [])
    }

    rows = []

    for zip_path in sorted(OUT_DIR.glob("*.zip")):
        match = re.search(r"_(\d+)\.zip$", zip_path.name)
        run_id = match.group(1) if match else "unknown"
        run = run_map.get(run_id, {})

        best = None

        try:
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    if member.endswith("/"):
                        continue

                    text = zf.read(member).decode("utf-8", errors="replace")
                    reason_type, reason, context = find_best_reason(text)

                    score = 0
                    for candidate_score, candidate_type, _ in REASON_PATTERNS:
                        if candidate_type == reason_type:
                            score = candidate_score
                            break

                    candidate = {
                        "score": score,
                        "run_number": run.get("run_number", ""),
                        "run_id": run_id,
                        "title": (
                            run.get("display_title")
                            or run.get("head_commit", {}).get("message")
                            or run.get("name")
                            or ""
                        ),
                        "created_at": run.get("created_at", ""),
                        "commit": (run.get("head_sha") or "")[:7],
                        "reason_type": reason_type,
                        "reason": reason,
                        "log_file": member,
                        "url": run.get("html_url", ""),
                        "context": context,
                    }

                    if best is None or candidate["score"] > best["score"]:
                        best = candidate

        except zipfile.BadZipFile:
            best = {
                "score": 0,
                "run_number": run.get("run_number", ""),
                "run_id": run_id,
                "title": (
                    run.get("display_title")
                    or run.get("head_commit", {}).get("message")
                    or run.get("name")
                    or ""
                ),
                "created_at": run.get("created_at", ""),
                "commit": (run.get("head_sha") or "")[:7],
                "reason_type": "bad_zip",
                "reason": "Downloaded log file is not a valid ZIP",
                "log_file": zip_path.name,
                "url": run.get("html_url", ""),
                "context": "",
            }

        if best is not None:
            rows.append(best)

    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)

    return rows


def write_outputs(rows: list[dict]) -> None:
    with SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# GitHub Actions Failure Summary\n\n")

        for row in rows:
            f.write(f"## Run #{row['run_number']} — {row['title']}\n\n")
            f.write(f"- Run ID: `{row['run_id']}`\n")
            f.write(f"- Commit: `{row['commit']}`\n")
            f.write(f"- Created: `{row['created_at']}`\n")
            f.write(f"- Reason type: `{row['reason_type']}`\n")
            f.write(f"- Failure reason: `{row['reason']}`\n")
            f.write(f"- Log file: `{row['log_file']}`\n")
            f.write(f"- URL: {row['url']}\n\n")
            f.write("```text\n")
            f.write(row["context"][:3500])
            f.write("\n```\n\n")

    with SUMMARY_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_number",
                "title",
                "run_id",
                "created_at",
                "commit",
                "reason_type",
                "reason",
                "log_file",
                "url",
            ],
            delimiter="\t",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "run_number": row["run_number"],
                "title": row["title"],
                "run_id": row["run_id"],
                "created_at": row["created_at"],
                "commit": row["commit"],
                "reason_type": row["reason_type"],
                "reason": row["reason"],
                "log_file": row["log_file"],
                "url": row["url"],
            })


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit(
            "ERROR: GITHUB_TOKEN is not set.\n"
            "Run:\n"
            "  read -s -p 'GitHub token: ' GITHUB_TOKEN\n"
            "  export GITHUB_TOKEN\n"
        )

    runs_data = fetch_all_runs(token)
    total = runs_data.get("total_count", 0)
    returned = len(runs_data.get("workflow_runs", []))
    failed = sum(
        1 for r in runs_data.get("workflow_runs", [])
        if r.get("conclusion") == "failure"
    )

    print()
    print("Run summary:")
    print(f"  total_count: {total}")
    print(f"  returned:    {returned}")
    print(f"  failed:      {failed}")
    print()

    download_failed_logs(runs_data, token)

    downloaded = len(list(OUT_DIR.glob("*.zip")))
    print()
    print(f"Downloaded ZIP logs available: {downloaded}")
    print("Summarizing logs...")

    rows = summarize_logs(runs_data)
    write_outputs(rows)

    print()
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {SUMMARY_TSV}")
    print(f"Summarized downloaded logs: {len(rows)}")

    if downloaded < failed:
        print()
        print("WARNING:")
        print(f"  GitHub reports {failed} failed runs, but only {downloaded} ZIP logs are available.")
        print("  Check download_errors.tsv if it exists.")


if __name__ == "__main__":
    main()
