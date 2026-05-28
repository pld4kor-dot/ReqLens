#!/usr/bin/env python3
"""Export demo artifacts (SRS, knowledge graph, metrics) for a project.

Usage:
    python -m scripts.export_demo_artifacts --project-id <PID> [--output-dir exports/]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def main(project_id: str, base_url: str, output_dir: Path) -> None:
    client = httpx.Client(base_url=base_url, timeout=120.0)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export SRS as Markdown
    print("Exporting SRS …")
    resp = client.get(f"/projects/{project_id}/export/srs", params={"format": "markdown"})
    if resp.status_code == 200:
        srs_path = output_dir / "generated_srs.md"
        srs_path.write_text(resp.text, encoding="utf-8")
        print(f"  ? {srs_path}")
    else:
        print(f"  SKIP SRS export ({resp.status_code})")

    # 2. Export requirements as JSON
    print("Exporting requirements …")
    resp = client.get(f"/projects/{project_id}/requirements")
    if resp.status_code == 200:
        reqs_path = output_dir / "requirements.json"
        reqs_path.write_text(json.dumps(resp.json(), indent=2, default=str), encoding="utf-8")
        print(f"  ? {reqs_path} ({len(resp.json())} requirements)")
    else:
        print(f"  SKIP requirements export ({resp.status_code})")

    # 3. Export knowledge graph
    print("Exporting knowledge graph …")
    resp = client.get(f"/projects/{project_id}/export/graph/json")
    if resp.status_code == 200:
        graph_path = output_dir / "knowledge_graph.json"
        graph_path.write_text(json.dumps(resp.json(), indent=2, default=str), encoding="utf-8")
        print(f"  ? {graph_path}")
    else:
        print(f"  SKIP graph export ({resp.status_code})")

    # 4. Export ReqIF (if available)
    print("Exporting ReqIF …")
    resp = client.get(f"/projects/{project_id}/export/reqif")
    if resp.status_code == 200:
        reqif_path = output_dir / "export.reqif"
        reqif_path.write_bytes(resp.content)
        print(f"  ? {reqif_path}")
    else:
        print(f"  SKIP ReqIF export ({resp.status_code})")

    # 5. Export traceability matrix
    print("Exporting traceability matrix …")
    resp = client.get(f"/projects/{project_id}/export/traceability")
    if resp.status_code == 200:
        trace_path = output_dir / "traceability_matrix.csv"
        trace_path.write_text(resp.text, encoding="utf-8")
        print(f"  ? {trace_path}")
    else:
        print(f"  SKIP traceability export ({resp.status_code})")

    print(f"\nAll artifacts saved to {output_dir.resolve()}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Export ReqInOne 2 demo artifacts.")
    parser.add_argument("--project-id", required=True, help="Project UUID")
    parser.add_argument("--base-url", default="http://localhost:8001", help="API base URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Output directory for exported files",
    )
    args = parser.parse_args()
    try:
        main(args.project_id, args.base_url, args.output_dir)
    except httpx.ConnectError:
        print("Could not connect to API. Is the server running?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
 