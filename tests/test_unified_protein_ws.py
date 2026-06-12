from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


API_ROOT = Path(__file__).resolve().parents[1] / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


class FakeModel:
    def ensure_ready(self) -> None:
        return None

    def generate(self, messages, **kwargs):
        del messages, kwargs
        return json.dumps(
            {
                "needs_clarification": True,
                "message": "Cau hoi cua ban chua ro. Minh co the chinh lai truy van.",
                "proposed_query": (
                    "plant growth-promoting protein crop yield nitrogen fixation "
                    "phosphate solubilization auxin biosynthesis ACC deaminase"
                ),
                "research_query": "plant growth-promoting proteins crop yield",
            }
        )


class FakeSearchResult:
    def to_dict(self):
        return {
            "title": "Plant growth-promoting proteins",
            "url": "https://example.com/pgpr",
            "published_date": "2026-01-02",
            "text": "Nitrogen fixation phosphate solubilization and auxin support crop yield.",
        }


class FakeSearchTool:
    def search(self, query):
        self.query = query
        return [FakeSearchResult()]


class FakeRecord:
    def __init__(self, accession, description, sequence, metadata):
        self.accession = accession
        self.source_name = "test"
        self.description = description
        self.organism = "Bacillus testis"
        self.sequence = sequence
        self.sequence_length = len(sequence)
        self.sequence_version = None
        self.metadata = metadata


class UnifiedProteinWebSocketTests(unittest.TestCase):
    def setUp(self) -> None:
        import server

        server._MODEL_CACHE.clear()
        server._TOOL_CACHE.clear()
        server._INFERENCE_CACHE.clear()

    def test_websocket_clarifies_then_semantic_ranks_and_returns_prompt(self) -> None:
        from fastapi.testclient import TestClient
        from server import PROTEIN_SPAN_COMPLETION_WS_ROUTE, create_app

        records = [
            FakeRecord(
                "BAD001",
                "uncharacterized membrane protein",
                "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEENFKALVLIAFAQYLQQC",
                {"product": "uncharacterized protein"},
            ),
            FakeRecord(
                "GOOD001",
                "nitrogenase iron protein for plant growth promotion",
                (
                    "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEENFKALVLIAFAQYLQQC"
                    "PFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMAD"
                    "CCAKQEPERNECFLSHKDDSPDLPK"
                ),
                {
                    "gene": "nifH",
                    "product": "nitrogenase iron protein",
                    "keywords": "plant growth-promoting rhizobacteria crop yield",
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "server._create_model",
            return_value=FakeModel(),
        ), patch(
            "server.ExaSearchTool",
            return_value=FakeSearchTool(),
        ), patch(
            "server._fetch_sequence_records",
            return_value=records,
        ):
            root = Path(temp_dir)
            protein_config = _write_api_config(root)
            agent_config = _write_agent_config(root)
            client = TestClient(
                create_app(
                    config_path=protein_config,
                    agent_config_path=agent_config,
                    environment="local",
                )
            )

            with client.websocket_connect(PROTEIN_SPAN_COMPLETION_WS_ROUTE) as websocket:
                websocket.send_json(
                    {
                        "user_input": "toi muon tang nang suat cay trong",
                        "limit": 2,
                        "mask_length": 12,
                    }
                )
                events = [websocket.receive_json()]
                while events[-1]["event"] != "waiting_for_user":
                    events.append(websocket.receive_json())

                websocket.send_json({"action": "approve"})
                while events[-1]["event"] not in {"completed", "error"}:
                    events.append(websocket.receive_json())

        event_names = [event["event"] for event in events]
        self.assertIn("clarification_completed", event_names)
        self.assertIn("public_research_completed", event_names)
        fetch_started = next(event for event in events if event["event"] == "fetch_started")
        self.assertEqual("ncbi", fetch_started["source"])
        self.assertIn("semantic_search_completed", event_names)
        completed = events[-1]
        self.assertEqual("completed", completed["event"], events)
        self.assertIn("instruction", completed)
        self.assertIn("input", completed)
        self.assertEqual("GOOD001", completed["selected_record"]["accession"])
        self.assertIn("<MASK_12>", completed["input"])


def _write_api_config(root: Path) -> Path:
    config_path = root / "config.yaml"
    config_path.write_text(
        f"""
environment: local

environments:
  local:
    model:
      path: {root.as_posix()}/model.onnx
      device: cpu
    server:
      host: 127.0.0.1
      port: 8000
      reload: false
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def _write_agent_config(root: Path) -> Path:
    config_path = root / "agent.yaml"
    config_path.write_text(
        """
environment: local

environments:
  local:
    openai:
      model: test-openai-model
      api_key_env: OPENAI_API_KEY
    exa:
      api_key_env: EXA_API_KEY
      search_type: neural
      max_results: 2
    agent:
      require_human_approval: false
      max_tool_calls: 1
    prompts:
      system_prompt_key: grounded_agent
  production:
    openai:
      model: test-openai-model
      api_key_env: OPENAI_API_KEY
    exa:
      api_key_env: EXA_API_KEY
      search_type: neural
      max_results: 2
    agent:
      require_human_approval: true
      max_tool_calls: 1
    prompts:
      system_prompt_key: grounded_agent
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


if __name__ == "__main__":
    unittest.main()
