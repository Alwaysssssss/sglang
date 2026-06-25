import time
import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, StrictFloat, StrictInt


# Image API protocol models
class ImageResponseData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None
    file_path: Optional[str] = None


class ImageResponse(BaseModel):
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageResponseData]
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None


class ImageGenerationsRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: Optional[int] = 1
    quality: Optional[str] = "auto"
    response_format: Optional[str] = "url"  # url | b64_json
    size: Optional[str] = "1024x1024"  # e.g., 1024x1024
    style: Optional[str] = "vivid"
    background: Optional[str] = "auto"  # transparent | opaque | auto
    output_format: Optional[str] = None  # png | jpeg | webp
    user: Optional[str] = None
    # SGLang extensions
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    true_cfg_scale: Optional[float] = (
        None  # for CFG vs guidance distillation (e.g., QwenImage)
    )
    seed: Optional[int] = 1024
    generator_device: Optional[str] = "cuda"
    negative_prompt: Optional[str] = None
    output_quality: Optional[str] = "default"
    output_compression: Optional[int] = None
    enable_teacache: Optional[bool] = False
    # Upscaling
    enable_upscaling: Optional[bool] = False
    upscaling_model_path: Optional[str] = None
    upscaling_scale: Optional[int] = 4
    diffusers_kwargs: Optional[Dict[str, Any]] = None  # kwargs for diffusers backend
    # Performance profiling
    perf_dump_path: Optional[str] = None


# Video API protocol models
class VideoResponse(BaseModel):
    id: str
    object: str = "video"
    model: str = "sora-2"
    status: str = "queued"
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    size: str = ""
    seconds: str = "4"
    quality: str = "standard"
    url: Optional[str] = None
    remixed_from_video_id: Optional[str] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    file_path: Optional[str] = None
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None


class VideoGenerationsRequest(BaseModel):
    prompt: str
    input_reference: Optional[str] = None
    reference_url: Optional[str] = None
    model: Optional[str] = None
    seconds: Optional[int] = 4
    size: Optional[str] = ""
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    seed: Optional[int] = 1024
    generator_device: Optional[str] = "cuda"
    # SGLang extensions
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None
    true_cfg_scale: Optional[float] = (
        None  # for CFG vs guidance distillation (e.g., QwenImage)
    )
    negative_prompt: Optional[str] = None
    enable_teacache: Optional[bool] = False
    # Frame interpolation
    enable_frame_interpolation: Optional[bool] = False
    frame_interpolation_exp: Optional[int] = 1  # 1=2×, 2=4×
    frame_interpolation_scale: Optional[float] = 1.0
    frame_interpolation_model_path: Optional[str] = None
    # Upscaling
    enable_upscaling: Optional[bool] = False
    upscaling_model_path: Optional[str] = None
    upscaling_scale: Optional[int] = 4
    output_quality: Optional[str] = "default"
    output_compression: Optional[int] = None
    output_path: Optional[str] = None
    diffusers_kwargs: Optional[Dict[str, Any]] = None  # kwargs for diffusers backend
    # Performance profiling
    perf_dump_path: Optional[str] = None


class VideoRepairMinioConfig(BaseModel):
    endpoint: str
    bucket_name: str
    access_key: str
    secret_key: str
    secure: bool = False
    region: str = "us-east-1"


def default_video_repair_output_object_key(
    request_id: str, now: datetime | None = None, extension: str = ".mp4"
) -> str:
    now = now or datetime.now()
    extension = extension if extension.startswith(".") else f".{extension}"
    return f"{now:%Y/%m/%d}/{now:%H%M%S}_{request_id}{extension}"


class VideoRepairRequest(BaseModel):
    task_id: Optional[str] = None
    timeout: int = -1
    prompt: str
    negative_prompt: Optional[str] = None
    model: Optional[str] = None

    video_input_path: Optional[str] = None
    mask_input_path: Optional[str] = None
    video_url: Optional[str] = None
    mask_url: Optional[str] = None
    reference_image_url: Optional[str] = None

    callback_url: Optional[str] = None
    minio_config: Optional[VideoRepairMinioConfig] = None
    output_storage: str = "local"
    output_path: Optional[str] = None
    output_bucket: Optional[str] = None
    output_object_key: Optional[str] = None

    num_frames: int = -1
    infer_len: int = 81
    overlap: int = 9
    strength: float = 1.0
    num_inference_steps: int = 40
    guidance_scale: float = 5.0
    seed: int = 42
    generator_device: Optional[str] = None
    dtype: str = "bf16"
    dynamic_cfg: bool = True
    dynamic_cfg_max_step: int = 15
    dynamic_cfg_min: float = 1.0

    bbox_padding: int = 0
    bbox_expand_scale: float = 1.2
    dilate_px: int = 15
    mask_scale: float = 1.2
    feather_px: int = 15
    adain_boundary_dilate: int = 0
    enable_paste_back: bool = True
    save_crop_only: bool = False
    drop_reference_frame: Optional[bool] = None
    keep_intermediate_windows: bool = False
    use_clip: bool = True
    use_repaired_context: bool = False
    vary_seed_by_window: bool = False
    init_latent_mode: Literal["noise", "add_noise"] = "noise"
    mask_downsample_mode: Literal["nearest", "nearest-exact"] = "nearest"
    overlap_commit_mode: Literal["native_skip", "weighted"] = "weighted"
    tail_padding_mode: Literal["native_reverse_mirror", "reflect"] = "reflect"
    decode_mode: Literal["eager", "stream"] = "stream"
    enable_teacache: bool = True
    teacache_thresh: float = 0.3
    teacache_start_skipping: Union[StrictInt, StrictFloat] = 5
    teacache_end_skipping: Union[StrictInt, StrictFloat] = 1.0
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


class VideoListResponse(BaseModel):
    data: List[VideoResponse]
    object: str = "list"


class VideoRemixRequest(BaseModel):
    prompt: str


# Mesh API protocol models
class MeshResponse(BaseModel):
    id: str
    object: str = "mesh"
    model: str = ""
    status: str = "queued"
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    format: str = "glb"
    url: Optional[str] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None


class MeshGenerationsRequest(BaseModel):
    prompt: str = "generate 3d mesh"
    input_image: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[int] = None
    generator_device: Optional[str] = "cuda"
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[str] = None
    output_format: Optional[str] = "glb"


class MeshListResponse(BaseModel):
    data: List[MeshResponse]
    object: str = "list"


@dataclass
class BaseReq(ABC):
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid


@dataclass
class VertexGenerateReqInput(BaseReq):
    instances: List[dict]
    parameters: Optional[dict] = None
