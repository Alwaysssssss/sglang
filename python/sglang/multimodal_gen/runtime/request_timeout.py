# SPDX-License-Identifier: Apache-2.0
import os
import time
from typing import Any

TASK_TIMEOUT_MESSAGE = "Request timed out."


class TaskTimeoutError(TimeoutError):
    pass


def request_timeout_deadline(timeout: float | int | None) -> float | None:
    if timeout is None:
        return None
    timeout = float(timeout)
    if timeout <= 0:
        return None
    return time.monotonic() + timeout


def remaining_request_timeout(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, float(deadline) - time.monotonic())


def check_request_timeout(request: Any) -> None:
    sampling_params = getattr(request, "sampling_params", None)
    cancel_path = getattr(request, "request_cancel_path", None)
    if cancel_path is None:
        cancel_path = getattr(sampling_params, "request_cancel_path", None)
    if cancel_path and os.path.exists(cancel_path):
        raise TaskTimeoutError(TASK_TIMEOUT_MESSAGE)

    deadline = getattr(request, "request_timeout_deadline", None)
    if deadline is None:
        deadline = getattr(sampling_params, "request_timeout_deadline", None)
    if deadline is not None and remaining_request_timeout(deadline) <= 0:
        raise TaskTimeoutError(TASK_TIMEOUT_MESSAGE)


def is_task_timeout_error(error: BaseException | str | None) -> bool:
    if error is None:
        return False
    if isinstance(error, TaskTimeoutError):
        return True
    return TASK_TIMEOUT_MESSAGE.lower() in str(error).lower()
