from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    context: str | list[str] | dict[str, Any] | None = None


class ApprovalRequest(BaseModel):
    approval_id: str = Field(..., min_length=1)


class RejectionRequest(ApprovalRequest):
    reason: str | None = None
    revised_instruction: str | None = None


class AgentRunResponse(BaseModel):
    status: Literal["completed", "waiting_for_human", "failed"]
    answer: str
    citations: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    needs_approval: bool = False
    approval_id: str | None = None
    error: str | None = None
