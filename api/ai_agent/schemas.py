from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    context: str | list[str] | dict[str, Any] | None = None


class ProteinSpanStartRequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    context: str | list[str] | dict[str, Any] | None = None
    source: Literal["ncbi", "ena", "auto"] = "ncbi"
    limit: int = Field(default=5, gt=0)
    semantic_top_k: int = Field(default=3, gt=0)
    mask_policy: str = "random_span"
    mask_start: int | None = Field(default=None, ge=0)
    mask_length: int = Field(default=48, gt=0)
    left_flank_size: int = Field(default=64, ge=0)
    right_flank_size: int = Field(default=64, ge=0)
    require_clarification: bool = True
    research_with_exa: bool = True


class ProteinSpanResumeRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    action: Literal["approve", "revise", "cancel"]
    user_input: str | None = None


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
