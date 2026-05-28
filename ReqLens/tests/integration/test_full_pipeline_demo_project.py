"""Integration test – full pipeline on demo project (requires Azure OpenAI)."""

import pytest


@pytest.mark.integration
@pytest.mark.llm
class TestFullPipelineDemoProject:
    """End-to-end pipeline test.

    Requires:
      - Azure OpenAI credentials in .env
      - PostgreSQL running (or SQLite for test mode)

    Run with: pytest -m integration
    """

    def test_placeholder(self):
        """Placeholder – implement when agents are connected to real LLM."""
        # Steps:
        # 1. Create project
        # 2. Ingest demo transcript
        # 3. Run extraction agent
        # 4. Run evidence agent
        # 5. Assert hallucination is blocked
        # 6. Run classification agent
        # 7. Run dependency agent
        # 8. Run composer agent
        # 9. Assert SRS contains only accepted requirements
        pass
