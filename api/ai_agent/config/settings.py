from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent
AI_AGENT_ROOT = CONFIG_DIR.parent
API_ROOT = AI_AGENT_ROOT.parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "agent.yaml"
DEFAULT_PROMPTS_PATH = CONFIG_DIR / "prompts.yaml"
DEFAULT_DOTENV_PATH = API_ROOT / ".env"
AGENT_CONFIG_PATH_ENV_VAR = "MDNAC_AGENT_CONFIG"
AGENT_ENVIRONMENT_ENV_VAR = "MDNAC_AGENT_ENV"
SUPPORTED_ENVIRONMENTS = {"local", "production"}


@dataclass(frozen=True)
class OpenAISettings:
    model: str
    api_key_env: str


@dataclass(frozen=True)
class ExaSettings:
    api_key_env: str
    search_type: str
    max_results: int


@dataclass(frozen=True)
class AgentSettings:
    require_human_approval: bool
    max_tool_calls: int


@dataclass(frozen=True)
class PromptSettings:
    system_prompt_key: str
    system_prompts: dict[str, str]


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    reload: bool


@dataclass(frozen=True)
class AISettings:
    environment: str
    config_path: Path
    prompts_path: Path
    openai: OpenAISettings
    exa: ExaSettings
    agent: AgentSettings
    prompts: PromptSettings
    server: ServerSettings

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def system_prompt(self) -> str:
        try:
            return self.prompts.system_prompts[self.prompts.system_prompt_key]
        except KeyError as exc:
            available = ", ".join(sorted(self.prompts.system_prompts))
            raise ValueError(
                "Unknown system prompt key "
                f"{self.prompts.system_prompt_key!r}. Available: {available}"
            ) from exc

    def require_openai_api_key(self) -> str:
        return _required_env_value(self.openai.api_key_env, service_name="OpenAI")

    def require_exa_api_key(self) -> str:
        return _required_env_value(self.exa.api_key_env, service_name="Exa")


def load_settings(
    *,
    config_path: str | Path | None = None,
    environment: str | None = None,
    prompts_path: str | Path | None = None,
) -> AISettings:
    _load_dotenv(DEFAULT_DOTENV_PATH)

    resolved_config_path = Path(
        config_path or os.environ.get(AGENT_CONFIG_PATH_ENV_VAR) or DEFAULT_CONFIG_PATH
    ).expanduser()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"AI agent config not found: {resolved_config_path}")

    resolved_prompts_path = Path(prompts_path or DEFAULT_PROMPTS_PATH).expanduser()
    if not resolved_prompts_path.is_file():
        raise FileNotFoundError(f"AI agent prompts config not found: {resolved_prompts_path}")

    payload = _read_yaml(resolved_config_path)
    prompt_payload = _read_yaml(resolved_prompts_path)
    environments = _required_mapping(payload, "environments", path=resolved_config_path)
    selected_environment = str(
        environment
        or os.environ.get(AGENT_ENVIRONMENT_ENV_VAR)
        or payload.get("environment")
        or "local"
    ).strip()
    if selected_environment not in SUPPORTED_ENVIRONMENTS:
        supported = ", ".join(sorted(SUPPORTED_ENVIRONMENTS))
        raise ValueError(
            f"Unsupported AI agent environment {selected_environment!r}. "
            f"Supported: {supported}"
        )
    if selected_environment not in environments:
        available = ", ".join(sorted(str(name) for name in environments))
        raise ValueError(
            f"Environment {selected_environment!r} missing from agent config. "
            f"Available: {available}"
        )

    raw_environment = environments[selected_environment]
    if not isinstance(raw_environment, dict):
        raise ValueError(f"Environment {selected_environment!r} must be a mapping.")

    prompts = _load_prompt_settings(raw_environment, prompt_payload)
    return AISettings(
        environment=selected_environment,
        config_path=resolved_config_path.resolve(),
        prompts_path=resolved_prompts_path.resolve(),
        openai=_load_openai_settings(raw_environment),
        exa=_load_exa_settings(raw_environment),
        agent=_load_agent_settings(raw_environment),
        prompts=prompts,
        server=_load_server_settings(raw_environment),
    )


def _load_openai_settings(raw_environment: dict[str, Any]) -> OpenAISettings:
    raw_openai = _required_mapping(raw_environment, "openai")
    return OpenAISettings(
        model=str(_required_value(raw_openai, "model")).strip(),
        api_key_env=str(_required_value(raw_openai, "api_key_env")).strip(),
    )


def _load_exa_settings(raw_environment: dict[str, Any]) -> ExaSettings:
    raw_exa = _required_mapping(raw_environment, "exa")
    return ExaSettings(
        api_key_env=str(_required_value(raw_exa, "api_key_env")).strip(),
        search_type=str(_required_value(raw_exa, "search_type")).strip(),
        max_results=int(_required_value(raw_exa, "max_results")),
    )


def _load_agent_settings(raw_environment: dict[str, Any]) -> AgentSettings:
    raw_agent = _required_mapping(raw_environment, "agent")
    return AgentSettings(
        require_human_approval=bool(_required_value(raw_agent, "require_human_approval")),
        max_tool_calls=int(_required_value(raw_agent, "max_tool_calls")),
    )


def _load_prompt_settings(
    raw_environment: dict[str, Any],
    prompt_payload: dict[str, Any],
) -> PromptSettings:
    raw_prompt_settings = _required_mapping(raw_environment, "prompts")
    raw_system_prompts = _required_mapping(prompt_payload, "system_prompts")
    system_prompts = {str(key): str(value) for key, value in raw_system_prompts.items()}
    return PromptSettings(
        system_prompt_key=str(_required_value(raw_prompt_settings, "system_prompt_key")).strip(),
        system_prompts=system_prompts,
    )


def _load_server_settings(raw_environment: dict[str, Any]) -> ServerSettings:
    raw_server = raw_environment.get("server") or {}
    if not isinstance(raw_server, dict):
        raise ValueError("Environment `server` must be a mapping when provided.")
    return ServerSettings(
        host=str(raw_server.get("host") or "127.0.0.1"),
        port=int(raw_server.get("port") or 8010),
        reload=bool(raw_server.get("reload", False)),
    )


def _required_mapping(
    payload: dict[str, Any],
    key: str,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        location = f" in {path}" if path else ""
        raise ValueError(f"Expected `{key}` mapping{location}.")
    return value


def _required_value(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required AI agent config field `{key}`.")
    return value


def _required_env_value(env_name: str, *, service_name: str) -> str:
    value = os.environ.get(env_name)
    if not value:
        raise ValueError(
            f"Missing {service_name} API key. Set environment variable `{env_name}`."
        )
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("AI agent config loading requires PyYAML.") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}.")
    return payload


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("'\"")
        if name and name not in os.environ:
            os.environ[name] = value


__all__ = [
    "AGENT_CONFIG_PATH_ENV_VAR",
    "AGENT_ENVIRONMENT_ENV_VAR",
    "AISettings",
    "AgentSettings",
    "ExaSettings",
    "OpenAISettings",
    "PromptSettings",
    "ServerSettings",
    "load_settings",
]
