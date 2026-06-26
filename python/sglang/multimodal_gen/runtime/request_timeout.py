import os
import time
from typing import Any

TASK_TIMEOUT_MESSAGE = "Request timed out."


class TaskTimeoutError(TimeoutError):
    pass


def request_timeout_deadline(timeout: int | float | None) -> float | None:
    normalized_timeout = 300 if timeout in (None, 0) else timeout
    if normalized_timeout == -1:
        return None
    return time.monotonic() + float(normalized_timeout)


def remaining_request_timeout(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return float(deadline) - time.monotonic()


def _maybe_get_attr(candidate: Any, name: str):
    if candidate is None:
        return None
    return getattr(candidate, name, None)


def check_request_timeout(request: Any) -> None:
    sampling_params = getattr(request, "sampling_params", None)
    cancel_path = _maybe_get_attr(request, "request_cancel_path") or _maybe_get_attr(
        sampling_params,
        "request_cancel_path",
    )
    if cancel_path and os.path.exists(cancel_path):
        raise TaskTimeoutError(TASK_TIMEOUT_MESSAGE)

    deadline = _maybe_get_attr(request, "request_timeout_deadline") or _maybe_get_attr(
        sampling_params,
        "request_timeout_deadline",
    )
    remaining = remaining_request_timeout(deadline)
    if remaining is not None and remaining <= 0:
        raise TaskTimeoutError(TASK_TIMEOUT_MESSAGE)


def is_task_timeout_error(error: BaseException | None) -> bool:
    current = error
    while current is not None:
        if isinstance(current, TaskTimeoutError):
            return True
        current = current.__cause__ or current.__context__
    return False
