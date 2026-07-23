# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base class for all pipeline executors.
"""

import contextlib
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler

if TYPE_CHECKING:
    # Only for type checkers; avoids runtime circular import
    from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage

logger = init_logger(__name__)


class Timer(StageProfiler):
    """
    A wrapper around StageProfiler to maintain backward compatibility.
    It forces simple logging behavior (log start/end) regardless of env vars.
    """

    def __init__(self, name="Stage"):
        super().__init__(
            stage_name=name, logger=logger, metrics=None, log_stage_start_end=True
        )


class PipelineExecutor(ABC):
    """
    Abstract base class for all pipeline executors.

    Executors orchestrate the execution of pipeline, with managing the parallel and communications required by stages

    """

    def __init__(self, server_args):
        self.server_args = server_args
        # GPUWorker sets RANK before constructing the pipeline. Cache that
        # process-local value so only one worker owns the process-global Kineto
        # profiler, even if distributed group rank lookup is not ready/stable.
        self.rank = int(os.environ.get("RANK", "0"))
        self._active_profiler = None

    def execute_with_profiling(
        self,
        stages: List["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:

        with self.profile_execution(batch, dump_rank=0):
            batch = self.execute(stages, batch, server_args)

        return batch

    @abstractmethod
    def execute(
        self,
        stages: List["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        Execute the pipeline stages.

        Args:
            stages: A list of pipeline stages to execute.
            batch: The batch to process.
            server_args: The server arguments.

        Returns:
            The processed batch.
        """
        raise NotImplementedError

    @contextlib.contextmanager
    def profile_execution(self, batch: Req, dump_rank: int = 0):
        """
        Context manager for profiling execution.
        """
        do_profile = batch.profile and not batch.is_warmup
        if not do_profile:
            # fast forward
            yield
            return

        request_id = batch.request_id
        rank = self.rank
        if rank != dump_rank:
            # Kineto is process-global in the diffusion worker topology. Starting
            # one profiler per distributed rank can race and corrupt its state.
            yield
            return

        # Some pipelines profile the whole request while reusing executor
        # helpers that also enter this context for individual windows. Kineto
        # does not support nested sessions in one process.
        if self._active_profiler is not None:
            yield
            return

        profiler = SGLDiffusionProfiler(
            request_id=request_id,
            rank=rank,
            full_profile=batch.profile_all_stages,
            num_steps=batch.num_profiled_timesteps,
            num_inference_steps=batch.num_inference_steps,
        )
        self._active_profiler = profiler
        try:
            yield
        finally:
            try:
                profiler.stop(dump_rank=dump_rank)
            finally:
                self._active_profiler = None
