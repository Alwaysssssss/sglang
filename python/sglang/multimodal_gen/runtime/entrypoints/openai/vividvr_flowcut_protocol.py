import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class VividVRFlowCutMinIOConfig(BaseModel):
    endpoint: str
    bucket_name: str = Field(alias="bucketName")
    access_key: str = Field(alias="accessKey")
    secret_key: str = Field(alias="secretKey")
    secure: bool = False
    region: Optional[str] = None

    model_config = {
        "populate_by_name": True,
        "extra": "forbid",
    }


class VividVRFlowCutRequest(BaseModel):
    task_id: Optional[str] = Field(default=None, alias="taskId")
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    model: Optional[str] = None

    video_input_path: Optional[str] = None
    video_url: Optional[str] = None
    caption_file_path: Optional[str] = None

    timeout: int = 300
    callback_url: Optional[str] = Field(default=None, alias="callbackUrl")
    minio_config: Optional[VividVRFlowCutMinIOConfig] = Field(
        default=None, alias="minioConfig"
    )
    output_storage: str = "local"
    output_path: Optional[str] = None
    output_bucket: Optional[str] = None
    output_object_key: Optional[str] = None

    num_frames: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: int = 42
    generator_device: Optional[str] = None
    dtype: str = "bf16"
    num_temporal_process_frames: Optional[int] = None
    restoration_guidance_scale: Optional[float] = None
    upscale: Optional[float] = None

    enable_teacache: bool = False
    enable_frame_interpolation: bool = False
    frame_interpolation_exp: int = 1
    frame_interpolation_scale: float = 1.0
    frame_interpolation_model_path: Optional[str] = None
    enable_upscaling: bool = False
    upscaling_model_path: Optional[str] = None
    upscaling_scale: int = 4
    output_quality: Optional[str] = "default"
    output_compression: Optional[int] = None
    perf_dump_path: Optional[str] = None

    @field_validator("timeout", mode="before")
    @classmethod
    def normalize_default_timeout(cls, value):
        if value is None or value == 0:
            return 300
        if int(value) < -1:
            raise ValueError("timeout must be positive or -1")
        return value

    @field_validator("upscale", mode="before")
    @classmethod
    def validate_original_vividvr_upscale_contract(cls, value):
        if value is None:
            return value
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                "upscale must be a finite number >= 0 that follows the original "
                f"Vivid-VR contract, got {value!r}"
            )
        return float(value)

    model_config = {
        "populate_by_name": True,
        "extra": "forbid",
    }


class VividVRFlowCutSubmitResponse(BaseModel):
    code: Literal[0, 1, 2]
    message: str = "ok"

    @field_validator("code", mode="before")
    @classmethod
    def reject_bool_code(cls, value):
        if isinstance(value, bool):
            raise ValueError("code must be numeric 0, 1, or 2")
        return value


class VividVRFlowCutCallbackOutput(BaseModel):
    result_url: str
    duration: Optional[float] = None

    model_config = {
        "extra": "forbid",
    }


class VividVRFlowCutCallbackPayload(BaseModel):
    status: Literal["running", "succeeded", "failed"]
    progress: float
    reason: str = ""
    output: str = ""
    success_output: Optional[VividVRFlowCutCallbackOutput] = Field(
        default=None, exclude=True
    )

    @classmethod
    def running(cls, *, progress: float, reason: str) -> "VividVRFlowCutCallbackPayload":
        return cls(
            status="running",
            progress=float(progress),
            reason=reason,
            output="",
        )

    @classmethod
    def succeeded(
        cls,
        *,
        result_url: str,
        duration: Optional[float] = None,
    ) -> "VividVRFlowCutCallbackPayload":
        success_output = VividVRFlowCutCallbackOutput(
            result_url=result_url,
            duration=duration,
        )
        return cls(
            status="succeeded",
            progress=100.0,
            reason="",
            output=success_output.model_dump_json(exclude_none=True),
            success_output=success_output,
        )

    @classmethod
    def failed(
        cls,
        *,
        reason: str,
        progress: float = 0.0,
    ) -> "VividVRFlowCutCallbackPayload":
        return cls(
            status="failed",
            progress=float(progress),
            reason=reason,
            output="",
        )


class VividVRFlowCutVideoResponse(BaseModel):
    id: str
    object: str = "video"
    model: str = "VividVR"
    status: str = "queued"
    progress: float = 0.0
    created_at: int
    size: str = ""
    seconds: str = ""
    quality: str = "standard"
    url: Optional[str] = None
    remixed_from_video_id: Optional[str] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    error: Optional[dict] = None
    reason: Optional[str] = None
    file_path: Optional[str] = None
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None


class VividVRFlowCutProgressResponse(BaseModel):
    id: str
    status: Optional[str] = None
    progress: float = 0.0
    file_path: Optional[str] = None
    url: Optional[str] = None
    error: Optional[dict] = None
    reason: Optional[str] = None
    callback_status: Optional[str] = None
    callback_error: Optional[str] = None
    callback_attempts: Optional[int] = None


FlowCutMinIOConfig = VividVRFlowCutMinIOConfig
FlowCutVideoRepairRequest = VividVRFlowCutRequest
FlowCutResponse = VividVRFlowCutSubmitResponse
FlowCutCallbackOutput = VividVRFlowCutCallbackOutput
FlowCutCallbackPayload = VividVRFlowCutCallbackPayload
FlowCutVideoResponse = VividVRFlowCutVideoResponse
FlowCutProgressResponse = VividVRFlowCutProgressResponse
