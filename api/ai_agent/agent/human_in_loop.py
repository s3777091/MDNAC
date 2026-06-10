from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class PendingApproval:
    approval_id: str
    draft_answer: str
    citations: list[str]
    tool_calls: list[dict[str, Any]]
    created_at: str
    user_input: str
    context: str


class InMemoryApprovalStore:
    def __init__(self) -> None:
        self._items: dict[str, PendingApproval] = {}
        self._lock = Lock()

    def save(
        self,
        *,
        draft_answer: str,
        citations: list[str],
        tool_calls: list[dict[str, Any]],
        user_input: str,
        context: str,
        approval_id: str | None = None,
    ) -> PendingApproval:
        item = PendingApproval(
            approval_id=approval_id or uuid4().hex,
            draft_answer=draft_answer,
            citations=citations,
            tool_calls=tool_calls,
            created_at=datetime.now(timezone.utc).isoformat(),
            user_input=user_input,
            context=context,
        )
        with self._lock:
            self._items[item.approval_id] = item
        return item

    def approve(self, approval_id: str) -> PendingApproval | None:
        with self._lock:
            return self._items.pop(approval_id, None)

    def reject(self, approval_id: str) -> PendingApproval | None:
        with self._lock:
            return self._items.pop(approval_id, None)

    def get(self, approval_id: str) -> PendingApproval | None:
        with self._lock:
            return self._items.get(approval_id)
