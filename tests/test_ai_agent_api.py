from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


API_ROOT = Path(__file__).resolve().parents[1] / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


class FakeModel:
    def __init__(self, answer: str = "Grounded answer.") -> None:
        self.answer = answer
        self.ready = False
        self.messages = []

    def ensure_ready(self) -> None:
        self.ready = True

    def generate(self, messages, **kwargs):
        del kwargs
        self.messages.append(messages)
        return self.answer


class FakeExaClient:
    def search_and_contents(self, query, **kwargs):
        del query, kwargs
        return {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com/research",
                    "publishedDate": "2026-01-02",
                    "text": "A concise result.",
                }
            ]
        }


class AIAgentAPITests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from ai_agent import server
        except ImportError:
            return
        server._MODEL_CACHE.clear()
        server._TOOL_CACHE.clear()

    def test_config_loads_local_environment(self) -> None:
        from ai_agent.config.settings import load_settings

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = _write_agent_config(Path(temp_dir), require_approval=False)
            settings = load_settings(config_path=config_path, environment="local")

        self.assertEqual("local", settings.environment)
        self.assertEqual("openai", settings.provider)
        self.assertEqual("test-openai-model", settings.openai.model)
        self.assertEqual("EXA_API_KEY", settings.exa.api_key_env)

    def test_health_works(self) -> None:
        from fastapi.testclient import TestClient
        from ai_agent.server import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = _write_agent_config(Path(temp_dir), require_approval=False)
            client = TestClient(create_app(config_path=config_path, environment="local"))

        response = client.get("/health")
        self.assertEqual(200, response.status_code)
        self.assertEqual("ok", response.json()["status"])

    def test_ready_returns_ok_with_mocked_model(self) -> None:
        from fastapi.testclient import TestClient
        from ai_agent.server import create_app

        fake_model = FakeModel()
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ai_agent.server._create_model",
            return_value=fake_model,
        ):
            config_path = _write_agent_config(Path(temp_dir), require_approval=False)
            client = TestClient(create_app(config_path=config_path, environment="local"))
            response = client.get("/ready")

        self.assertEqual(200, response.status_code, response.text)
        self.assertTrue(fake_model.ready)
        self.assertEqual("ready", response.json()["status"])

    def test_agent_skills_endpoint_lists_skill_markdown(self) -> None:
        from fastapi.testclient import TestClient
        from ai_agent.server import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = _write_agent_config(Path(temp_dir), require_approval=False)
            client = TestClient(create_app(config_path=config_path, environment="local"))
            response = client.get("/agent/skills")

        self.assertEqual(200, response.status_code, response.text)
        skill_names = {skill["name"] for skill in response.json()["skills"]}
        self.assertIn("grounded-answer", skill_names)
        self.assertIn("public-research", skill_names)
        self.assertIn("protein-span-completion", skill_names)

    def test_agent_run_returns_completed_without_human_approval(self) -> None:
        from fastapi.testclient import TestClient
        from ai_agent.server import create_app

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ai_agent.server._create_model",
            return_value=FakeModel("Answer from supplied context."),
        ):
            config_path = _write_agent_config(Path(temp_dir), require_approval=False)
            client = TestClient(create_app(config_path=config_path, environment="local"))
            response = client.post(
                "/agent/run",
                json={
                    "user_input": "Summarize this.",
                    "context": "The context contains the answer.",
                },
            )

        self.assertEqual(200, response.status_code, response.text)
        payload = response.json()
        self.assertEqual("completed", payload["status"])
        self.assertFalse(payload["needs_approval"])
        self.assertIsNone(payload["approval_id"])

    def test_agent_run_injects_selected_skills_into_model_prompt(self) -> None:
        from fastapi.testclient import TestClient
        from ai_agent.server import create_app

        fake_model = FakeModel("Skill-guided answer.")
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ai_agent.server._create_model",
            return_value=fake_model,
        ):
            config_path = _write_agent_config(Path(temp_dir), require_approval=False)
            client = TestClient(create_app(config_path=config_path, environment="local"))
            response = client.post(
                "/agent/run",
                json={
                    "user_input": "Create a protein span completion prompt for crop yield.",
                    "context": "Use the provided context only.",
                },
            )

        self.assertEqual(200, response.status_code, response.text)
        prompt_text = "\n".join(message["content"] for message in fake_model.messages[0])
        self.assertIn("Selected Skills:", prompt_text)
        self.assertIn("grounded-answer", prompt_text)
        self.assertIn("protein-span-completion", prompt_text)

    def test_agent_run_returns_waiting_for_human_when_required(self) -> None:
        from fastapi.testclient import TestClient
        from ai_agent.server import create_app

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ai_agent.server._create_model",
            return_value=FakeModel("Draft for review."),
        ):
            config_path = _write_agent_config(Path(temp_dir), require_approval=True)
            client = TestClient(create_app(config_path=config_path, environment="local"))
            response = client.post(
                "/agent/run",
                json={
                    "user_input": "Summarize this.",
                    "context": "The context contains the answer.",
                },
            )

        self.assertEqual(200, response.status_code, response.text)
        payload = response.json()
        self.assertEqual("waiting_for_human", payload["status"])
        self.assertTrue(payload["needs_approval"])
        self.assertTrue(payload["approval_id"])

    def test_approve_finalizes_pending_draft(self) -> None:
        from fastapi.testclient import TestClient
        from ai_agent.server import create_app

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ai_agent.server._create_model",
            return_value=FakeModel("Draft answer."),
        ):
            config_path = _write_agent_config(Path(temp_dir), require_approval=True)
            client = TestClient(create_app(config_path=config_path, environment="local"))
            run_response = client.post(
                "/agent/run",
                json={"user_input": "Answer this.", "context": "Known context."},
            )
            approval_id = run_response.json()["approval_id"]
            approve_response = client.post(
                "/agent/approve",
                json={"approval_id": approval_id},
            )

        self.assertEqual(200, approve_response.status_code, approve_response.text)
        payload = approve_response.json()
        self.assertEqual("completed", payload["status"])
        self.assertEqual("Draft answer.", payload["answer"])
        self.assertFalse(payload["needs_approval"])

    def test_exa_tool_normalizes_mocked_search_results(self) -> None:
        from ai_agent.config.settings import ExaSettings
        from ai_agent.tools.exa_search import ExaSearchTool

        settings = ExaSettings(
            api_key_env="EXA_API_KEY",
            search_type="neural",
            max_results=3,
        )
        with patch.dict(os.environ, {"EXA_API_KEY": "test-key"}):
            tool = ExaSearchTool(settings, client_factory=lambda api_key: FakeExaClient())
            results = tool.search("test query")

        self.assertEqual(1, len(results))
        self.assertEqual("Example", results[0].title)
        self.assertEqual("https://example.com/research", results[0].url)
        self.assertEqual("2026-01-02", results[0].published_date)
        self.assertEqual("A concise result.", results[0].text)

    def test_answer_response_does_not_include_fake_citations(self) -> None:
        from fastapi.testclient import TestClient
        from ai_agent.server import create_app

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ai_agent.server._create_model",
            return_value=FakeModel("Answer cites https://fake.example/source."),
        ):
            config_path = _write_agent_config(Path(temp_dir), require_approval=False)
            client = TestClient(create_app(config_path=config_path, environment="local"))
            response = client.post(
                "/agent/run",
                json={"user_input": "Summarize this.", "context": "Known context."},
            )

        self.assertEqual(200, response.status_code, response.text)
        payload = response.json()
        self.assertEqual([], payload["citations"])
        self.assertNotIn("https://fake.example/source", payload["answer"])


def _write_agent_config(root: Path, *, require_approval: bool) -> Path:
    config_path = root / "agent.yaml"
    config_path.write_text(
        f"""
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
      require_human_approval: {str(require_approval).lower()}
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
