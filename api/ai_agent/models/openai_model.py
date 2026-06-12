from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OpenAIChatModel:
    model: str
    api_key: str

    def __post_init__(self) -> None:
        self._client = None
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")

    def ensure_ready(self) -> None:
        self._get_client()

    def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        client = self._get_client()
        temperature = kwargs.get("temperature", 0.0)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as exc:
            # Newer model families (e.g. gpt-5.x) reject a non-default temperature
            # on chat.completions. Retry once with the provider default.
            if temperature is not None and self._is_temperature_error(exc):
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
            else:
                raise
        content = response.choices[0].message.content
        return str(content or "")

    @staticmethod
    def _is_temperature_error(exc: Exception) -> bool:
        message = str(getattr(exc, "message", "") or exc).lower()
        return "temperature" in message

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI provider requires the `openai` package. "
                    "Install the ai agent extra dependencies."
                ) from exc
            self._client = OpenAI(api_key=self.api_key)
        return self._client
