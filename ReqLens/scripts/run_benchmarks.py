#!/usr/bin/env python3
"""Run evaluation benchmarks and produce reports.

Usage:
    python -m scripts.run_benchmarks [--base-url http://localhost:8001] [--output-dir data/benchmarks]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx


def main(base_url: str, output_dir: Path) -> None:
    client = httpx.Client(base_url=base_url, timeout=600.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    print("Fetching projects ...")
    resp = client.get("/projects")
    resp.raise_for_status()
    projects = resp.json()

    if not projects:
        print("No projects found. Ingest a demo project first.", file=sys.stderr)
        sys.exit(1)

    all_results: list[dict] = []
    resp = client.post(
            f"/benchmarks/reqfromsrs/run",
            json={"benchmark_type": "reqfromsrs_classification", "dataset_type":"ReqFromSRS", "config":{}},
        )
    if resp.status_code != 200:
            print(f"  SKIP benchmark endpoint returned {resp.status_code}")
    result = resp.json()
    result["project_name"] = "reqfromsrs"
    result["timestamp"] = timestamp
    all_results.append(result)
    # for proj in projects:
    #     pid = proj["id"]
    #     name = proj.get("name", pid)
    #     print(f"\n=== Benchmarking: {name} ===")

    #     # Run benchmark endpoint
    #     resp = client.post(
    #         f"/benchmarks/{pid}/run",
    #         json={"suites": ["promise", "reqfromsrs", "hallucination", "dependency"]},
    #     )
    #     if resp.status_code != 200:
    #         print(f"  SKIP benchmark endpoint returned {resp.status_code}")
    #         continue

    #     result = resp.json()
    #     result["project_name"] = name
    #     result["timestamp"] = timestamp
    #     all_results.append(result)

    #     # Print summary
    #     for suite, metrics in result.get("suites", {}).items():
    #         print(f"  {suite}:")
    #         for k, v in metrics.items():
    #             if isinstance(v, float):
    #                 print(f"    {k}: {v:.4f}")
    #             else:
    #                 print(f"    {k}: {v}")

    # Save combined results
    out_file = output_dir / f"benchmark_{timestamp}.json"
    out_file.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {out_file}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run ReqInOne 2 evaluation benchmarks.")
    parser.add_argument("--base-url", default="http://localhost:8001", help="API base URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmarks"),
        help="Output directory for benchmark results",
    )
    args = parser.parse_args()
    try:
        main(args.base_url, args.output_dir)
    except httpx.ConnectError:
        print("Could not connect to API. Is the server running?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
 