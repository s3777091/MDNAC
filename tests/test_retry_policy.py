"""Unit tests for RetryPolicy."""

from __future__ import annotations

import unittest

from libs.data.utilities.retry import RetryPolicy


class RetryPolicyTests(unittest.TestCase):
    def test_executes_operation_without_retry_on_success(self) -> None:
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            return ["result"]

        policy = RetryPolicy(max_retries=3, backoff_base=0.0)
        result = policy.execute(operation=operation, is_empty=lambda r: not r, context="test")
        self.assertEqual(result, ["result"])
        self.assertEqual(call_count, 1)

    def test_retries_on_empty_result(self) -> None:
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return []
            return ["data"]

        policy = RetryPolicy(max_retries=5, backoff_base=0.0)
        result = policy.execute(operation=operation, is_empty=lambda r: not r, context="test")
        self.assertEqual(result, ["data"])
        self.assertEqual(call_count, 3)

    def test_returns_empty_after_exhausting_retries(self) -> None:
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            return []

        policy = RetryPolicy(max_retries=2, backoff_base=0.0)
        result = policy.execute(operation=operation, is_empty=lambda r: not r, context="test")
        self.assertEqual(result, [])
        self.assertEqual(call_count, 3)  # 1 initial + 2 retries


if __name__ == "__main__":
    unittest.main()
