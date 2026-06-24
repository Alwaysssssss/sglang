from dataclasses import dataclass

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client


@dataclass(frozen=True)
class VideoGenerationJobResult:
    save_file_path: str
    result: OutputBatch


async def run_video_generation_job(batch: Req) -> VideoGenerationJobResult:
    save_file_path_list, result = await process_generation_batch(
        async_scheduler_client, batch
    )
    return VideoGenerationJobResult(
        save_file_path=save_file_path_list[0],
        result=result,
    )
