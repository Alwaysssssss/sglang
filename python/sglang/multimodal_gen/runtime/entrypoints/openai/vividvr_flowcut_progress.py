from collections.abc import Awaitable, Callable
from typing import Any, Literal

from sglang.multimodal_gen.runtime.entrypoints.openai.flowcut import (
    post_flowcut_callback,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    VividVRFlowCutCallbackPayload,
)

VividVRFlowCutStage = Literal[
    "accepted",
    "input_ready",
    "caption_ready",
    "editing",
    "uploading_result",
    "succeeded",
    "failed",
]

FLOWCUT_STAGE_PROGRESS: dict[str, float] = {
    "accepted": 1.0,
    "input_ready": 10.0,
    "caption_ready": 20.0,
    "editing": 60.0,
    "uploading_result": 90.0,
    "succeeded": 100.0,
}

FLOWCUT_STAGE_REASONS: dict[str, str] = {
    "accepted": "accepted",
    "input_ready": "input_ready",
    "caption_ready": "caption_ready",
    "editing": "editing",
    "uploading_result": "uploading_result",
    "succeeded": "succeeded",
}

PostFlowCutCallback = Callable[
    [str, dict[str, Any]],
    Awaitable[None],
]


class VividVRFlowCutProgressReporter:
    """Stage-based FlowCut callback reporter for Vivid-VR repair jobs."""

    def __init__(
        self,
        task_id: str,
        callback_url: str,
        *,
        post_callback: PostFlowCutCallback = post_flowcut_callback,
    ) -> None:
        self.task_id = task_id
        self.callback_url = callback_url
        self._post_callback = post_callback
        self._last_progress = 0.0

    async def send_stage(self, stage: VividVRFlowCutStage) -> None:
        if stage not in FLOWCUT_STAGE_PROGRESS or stage == "succeeded":
            raise ValueError(f"Unsupported running FlowCut stage: {stage}")

        progress = FLOWCUT_STAGE_PROGRESS[stage]
        payload = VividVRFlowCutCallbackPayload.running(
            progress=progress,
            reason=FLOWCUT_STAGE_REASONS[stage],
        ).model_dump()
        self._last_progress = progress
        await self._post_callback(self.callback_url, payload)

    async def send_succeeded(
        self,
        result_url: str,
        *,
        duration: float | None = None,
    ) -> None:
        callback_payload = VividVRFlowCutCallbackPayload.succeeded(
            result_url=result_url,
            duration=duration,
        )
        payload = callback_payload.model_dump()
        payload["reason"] = FLOWCUT_STAGE_REASONS["succeeded"]
        self._last_progress = FLOWCUT_STAGE_PROGRESS["succeeded"]
        await self._post_callback(self.callback_url, payload)

    async def send_failed(self, reason: str, *, progress: float | None = None) -> None:
        failed_progress = self._last_progress if progress is None else float(progress)
        payload = VividVRFlowCutCallbackPayload.failed(
            reason=reason,
            progress=failed_progress,
        ).model_dump()
        await self._post_callback(self.callback_url, payload)
