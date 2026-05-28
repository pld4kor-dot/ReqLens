#!/usr/bin/env python3
"""Ingest the University Event Management System demo project.

Usage:
    python -m scripts.ingest_demo_project [--base-url http://localhost:8001]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "demo_projects" / "university_events"

FILES = [
    ("01_stakeholder_transcript.txt", "transcript"),
    ("02_legacy_srs.md", "srs"),
    ("03_user_stories.md", "user_story"),
    ("04_test_cases.md", "test_case"),
    ("05_security_policy.md", "policy"),
    ("06_change_request.md", "change_request"),
]


def main(base_url: str = "http://localhost:8001") -> None:
    client = httpx.Client(base_url=base_url, timeout=60.0)

    # 1. Create project
    print("Creating project …")
    resp = client.post(
        "/projects",
        json={
            "name": "College Event Management System",
            "description": (
                "A web-based platform for managing campus events – creation, "
                "discovery, registration, attendance tracking, and reporting."
            ),
        },
    )
    resp.raise_for_status()
    project = resp.json()
    project_id = project["id"]
    print(f"  Project created: {project_id}")

    # 2. Upload documents
    for filename, doc_type in FILES:
        filepath = DEMO_DIR / filename
        if not filepath.exists():
            print(f"  SKIP {filename} (not found)")
            continue
        print(f"  Uploading {filename} as {doc_type} …")
        resp = client.post(
            f"/projects/{project_id}/documents",
            files={"file": (filename, filepath.read_bytes(), "text/plain")},
            data={"doc_type": doc_type},
        )
        print("Got File")
        resp.raise_for_status()
        doc = resp.json()
        print(f"    Document created: {doc['id']}")

    print(f"\nDone. Project {project_id} ready for pipeline execution.")
    print(f"Run:  python -m scripts.run_pipeline --project-id {project_id}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Ingest demo project into ReqInOne 2.")
    parser.add_argument("--base-url", default="http://localhost:8001", help="API base URL")
    args = parser.parse_args()
    try:
        main(args.base_url)
    except httpx.HTTPStatusError as exc:
        print(f"HTTP error: {exc.response.status_code} – {exc.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print("Could not connect to API. Is the server running?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
