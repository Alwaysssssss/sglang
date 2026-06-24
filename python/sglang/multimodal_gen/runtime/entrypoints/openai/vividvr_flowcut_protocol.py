import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class VividVRFlowCutMinIOConfig(BaseModel):
    endpoint: str
    bucket_name: str
    access_key: str
    secret_key: str
    secure: bool = False
    region: Optional[str] = None


class VividVRFlowCutRequest(BaseModel):
    task_id: Optional[str] = Field(default=None, alias="taskId")
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    model: Optional[str] = None

    video_input_path: Optional[str] = None
    mask_input_path: Optional[str] = None
    video_url: Optional[str] = None
    mask_url: Optional[str] = None
    caption_file_path: Optional[str] = None
    reference_video_path: Optional[str] = None

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
    infer_len: int = 81
    overlap: int = 0
    strength: float = 1.0
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: int = 42
    generator_device: Optional[str] = None
    dtype: str = "bf16"
    num_temporal_process_frames: Optional[int] = None
    restoration_guidance_scale: Optional[float] = None
    dynamic_cfg: bool = True
    dynamic_cfg_max_step: int = 15
    dynamic_cfg_min: float = 1.0

    bbox_padding: int = 0
    dilate_px: int = 15
    mask_scale: float = 1.2
    feather_px: int = 12
    adain_boundary_dilate: int = 15
    enable_paste_back: bool = True
    save_crop_only: bool = False
    drop_reference_frame: bool = True
    keep_intermediate_windows: bool = False
    use_repaired_context: bool = True
    vary_seed_by_window: bool = False
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
        return value

    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }


class VividVRFlowCutSubmitResponse(BaseModel):
    code: Literal[0, 1, 2]
    message: str = "ok"


class VividVRFlowCutCallbackPayload(BaseModel):
    status: Literal["running", "succeeded", "failed"]
    progress: float
    reason: str = ""
    output: str = ""

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
        output = {"result_url": result_url}
        if duration is not None:
            output["duration"] = duration
        return cls(
            status="succeeded",
            progress=100.0,
            reason="",
            output=json.dumps(output, ensure_ascii=False),
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


FlowCutMinIOConfig = VividVRFlowCutMinIOConfig
FlowCutVideoRepairRequest = VividVRFlowCutRequest
FlowCutResponse = VividVRFlowCutSubmitResponse
FlowCutCallbackPayload = VividVRFlowCutCallbackPayload
