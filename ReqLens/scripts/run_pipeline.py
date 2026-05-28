#!/usr/bin/env python3
"""Run the full ReqInOne 2 pipeline for a project.

Usage:

    python -m scripts.run_pipeline --project-id <PROJECT_ID> [--base-url http://localhost:8001]

"""

from __future__ import annotations

import argparse
import sys
import time
import httpx

def main(project_id: str, base_url: str = "http://localhost:8001") -> None:
    """Triggers and monitors the full pipeline for a project."""
    client = httpx.Client(base_url=base_url, timeout=600.0)  # Increased timeout for full run

    print(f"Running full pipeline for project {project_id}...\n")
    t0 = time.perf_counter()

    try:
        # The server endpoint runs the full pipeline by default.
        # We pass `None` or an empty body to trigger all steps.
        resp = client.post(f"/projects/{project_id}/pipeline", json={})

        resp.raise_for_status()
        result = resp.json()

        elapsed = time.perf_counter() - t0
        status = result.get("status", "unknown")

        print(f"Pipeline status: {status} ({elapsed:.1f}s)")

        if status == "failed":
            print(f"  Error: {result.get('error', 'unknown')}", file=sys.stderr)
            sys.exit(1)
        
        # Print summary from the final response
        steps_run = result.get('steps_run', [])
        print(f"\nSteps executed: {', '.join(steps_run) if steps_run else 'None'}")


    except httpx.HTTPStatusError as exc:
        print(f"HTTP error: {exc.response.status_code} -- {exc.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print("Could not connect to API. Is the server running?", file=sys.stderr)
        sys.exit(1)


    # --- Final Summary ---
    # This part remains useful to verify the outcome.
    print("\n--- Final Project Summary ---")
    try:
        resp = client.get(f"/projects/{project_id}/requirements")
        resp.raise_for_status()
        reqs = resp.json()

        print(f"Total requirements extracted: {len(reqs)}")
        
        accepted = [r for r in reqs if r.get("status") == "accepted"]
        print(f"Accepted (SRS-eligible):      {len(accepted)}")
        print(f"Pending review:               {len(reqs) - len(accepted)}")

    except httpx.HTTPError as exc:
        print(f"Could not retrieve final summary: {exc}", file=sys.stderr)


def cli() -> None:
    """Command Line Interface setup."""
    parser = argparse.ArgumentParser(description="Run the ReqInOne 2 pipeline.")
    parser.add_argument("--project-id", required=True, help="Project UUID")
    parser.add_argument("--base-url", default="http://localhost:8001", help="API base URL")
    args = parser.parse_args()
    
    main(args.project_id, args.base_url)

if __name__ == "__main__":
    cli()
