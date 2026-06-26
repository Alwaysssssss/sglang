# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading

from PIL import Image

from sglang.multimodal_gen.runtime.videoedit.postprocess import paste_back_frame
from sglang.multimodal_gen.runtime.videoedit.preprocess import (
    crop_frame,
    expand_mask_frame,
    resize_frame,
)
from sglang.multimodal_gen.runtime.videoedit.stream_decoder import (
    SequentialMaskDecoder,
    SequentialVideoDecoder,
)


@dataclass
class _FrameEntry:
    index: int
    resized_frame: Image.Image
    resized_mask: Image.Image


class WindowFrameProvider:
    def __init__(
        self,
        *,
        video_input_path: str,
        mask_input_path: str,
        reference_image_path: str | None,
        num_frames: int,
        fps: float,
        frame_size: tuple[int, int],
        bbox: tuple[int, int, int, int],
        crop_h: int,
        crop_w: int,
        aligned_h: int,
        aligned_w: int,
        dilate_px: int,
        mask_scale: float,
        infer_len: int,
        enable_prefetch: bool = True,
        cache_max_frames: int | None = None,
        prefetch_ahead_frames: int | None = None,
    ):
        self.video_input_path = video_input_path
        self.mask_input_path = mask_input_path
        self.reference_image_path = reference_image_path
        self.num_frames = num_frames
        self.fps = fps
        self.frame_size = frame_size
        self.bbox = bbox
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.aligned_h = aligned_h
        self.aligned_w = aligned_w
        self.dilate_px = dilate_px
        self.mask_scale = mask_scale
        self.enable_prefetch = enable_prefetch
        self.cache_max_frames = cache_max_frames or max(infer_len * 2, infer_len + 8)
        self.prefetch_ahead_frames = prefetch_ahead_frames or infer_len
        self.reference_offset = 1 if reference_image_path else 0
        self._reference_frame = self._load_reference_frame()
        self._video_decoder = SequentialVideoDecoder(video_input_path)
        self._mask_decoder = SequentialMaskDecoder(mask_input_path, target_size=frame_size)
        self._next_decode_index = 0
        self._target_decode_index = -1
        self._cache: OrderedDict[int, _FrameEntry] = OrderedDict()
        self._decode_error: Exception | None = None
        self._closed = False
        self._cond = threading.Condition()
        self._prefetch_thread: threading.Thread | None = None
        if enable_prefetch:
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_loop,
                name="videoedit-frame-prefetch",
                daemon=True,
            )
            self._prefetch_thread.start()

    @classmethod
    def from_scanned_geometry(
        cls,
        *,
        video_input_path: str,
        mask_input_path: str,
        reference_image_path: str | None,
        scanned_geometry: dict,
        dilate_px: int,
        mask_scale: float,
        infer_len: int,
        enable_prefetch: bool = True,
    ) -> "WindowFrameProvider":
        return cls(
            video_input_path=video_input_path,
            mask_input_path=mask_input_path,
            reference_image_path=reference_image_path,
            num_frames=int(scanned_geometry["num_frames"]),
            fps=float(scanned_geometry["fps"]),
            frame_size=tuple(scanned_geometry["frame_size"]),
            bbox=tuple(scanned_geometry["bbox"]),
            crop_h=int(scanned_geometry["crop_h"]),
            crop_w=int(scanned_geometry["crop_w"]),
            aligned_h=int(scanned_geometry["aligned_h"]),
            aligned_w=int(scanned_geometry["aligned_w"]),
            dilate_px=dilate_px,
            mask_scale=mask_scale,
            infer_len=infer_len,
            enable_prefetch=enable_prefetch,
        )

    def _load_reference_frame(self) -> Image.Image | None:
        if self.reference_image_path is None:
            return None
        with Image.open(self.reference_image_path) as image:
            return image.convert("RGB").resize(self.frame_size)

    def _make_video_decoder(self) -> SequentialVideoDecoder:
        return SequentialVideoDecoder(self.video_input_path)

    def _make_mask_decoder(self) -> SequentialMaskDecoder:
        return SequentialMaskDecoder(self.mask_input_path, target_size=self.frame_size)

    def _prefetch_loop(self) -> None:
        try:
            while True:
                with self._cond:
                    while (
                        not self._closed
                        and (
                            self._next_decode_index >= self.num_frames
                            or self._next_decode_index > self._target_decode_index
                        )
                    ):
                        self._cond.wait()
                    if self._closed or self._next_decode_index >= self.num_frames:
                        return
                entry = self._decode_one(self._next_decode_index)
                with self._cond:
                    self._store_entry(entry)
                    self._next_decode_index += 1
                    self._cond.notify_all()
        except Exception as exc:  # pragma: no cover - exercised via waiters
            with self._cond:
                self._decode_error = exc
                self._cond.notify_all()

    def _store_entry(self, entry: _FrameEntry) -> None:
        self._cache[entry.index] = entry
        self._cache.move_to_end(entry.index)
        while len(self._cache) > self.cache_max_frames:
            self._cache.popitem(last=False)

    def _decode_one(self, global_index: int) -> _FrameEntry:
        if global_index == 0 and self.reference_offset == 1:
            assert self._reference_frame is not None
            frame = self._reference_frame.copy()
            raw_mask = Image.new("L", self.frame_size, 0)
        else:
            frame = self._video_decoder.read_next()
            raw_mask = self._mask_decoder.read_next()
            if frame is None or raw_mask is None:
                raise RuntimeError(
                    "Unexpected end of input while streaming VideoEdit frames "
                    f"at global index {global_index}"
                )
        expanded_mask = expand_mask_frame(
            raw_mask,
            dilate_px=self.dilate_px,
            scale=self.mask_scale,
            force_zero=global_index == 0,
        )
        cropped_frame = crop_frame(frame, self.bbox)
        cropped_mask = crop_frame(expanded_mask, self.bbox)
        return _FrameEntry(
            index=global_index,
            resized_frame=resize_frame(cropped_frame, self.aligned_h, self.aligned_w),
            resized_mask=resize_frame(cropped_mask, self.aligned_h, self.aligned_w),
        )

    def _ensure_decoded_through(self, index: int) -> None:
        if index < 0:
            return
        if self.enable_prefetch:
            self._wait_for_index(index)
            return
        self._decode_sync_until(index)

    def _decode_sync_until(self, index: int) -> None:
        while self._next_decode_index <= index:
            self._raise_if_error()
            entry = self._decode_one(self._next_decode_index)
            self._store_entry(entry)
            self._next_decode_index += 1

    def _wait_for_index(self, index: int) -> None:
        if index >= self.num_frames:
            raise IndexError(f"Frame index out of range: {index} >= {self.num_frames}")
        target = min(self.num_frames - 1, index + self.prefetch_ahead_frames)
        with self._cond:
            self._target_decode_index = max(self._target_decode_index, target)
            self._cond.notify_all()
            while self._next_decode_index <= index and self._decode_error is None and not self._closed:
                self._cond.wait()
            self._raise_if_error()

    def _reopen_decode_entry(self, index: int) -> _FrameEntry:
        video_decoder = self._make_video_decoder()
        mask_decoder = self._make_mask_decoder()
        try:
            reference_frame = self._reference_frame.copy() if self._reference_frame is not None else None
            for global_index in range(index + 1):
                if global_index == 0 and self.reference_offset == 1:
                    assert reference_frame is not None
                    frame = reference_frame
                    raw_mask = Image.new("L", self.frame_size, 0)
                else:
                    frame = video_decoder.read_next()
                    raw_mask = mask_decoder.read_next()
                    if frame is None or raw_mask is None:
                        raise RuntimeError(
                            f"Unexpected end of input while reopening frame {index}"
                        )
                if global_index == index:
                    expanded_mask = expand_mask_frame(
                        raw_mask,
                        dilate_px=self.dilate_px,
                        scale=self.mask_scale,
                        force_zero=global_index == 0,
                    )
                    cropped_frame = crop_frame(frame, self.bbox)
                    cropped_mask = crop_frame(expanded_mask, self.bbox)
                    return _FrameEntry(
                        index=index,
                        resized_frame=resize_frame(cropped_frame, self.aligned_h, self.aligned_w),
                        resized_mask=resize_frame(cropped_mask, self.aligned_h, self.aligned_w),
                    )
            raise RuntimeError(f"Could not reopen frame {index}")
        finally:
            video_decoder.close()
            mask_decoder.close()

    def _lookup_entry(self, index: int) -> _FrameEntry:
        entry = self._cache.get(index)
        if entry is not None:
            return entry
        return self._reopen_decode_entry(index)

    def _raise_if_error(self) -> None:
        if self._decode_error is not None:
            raise RuntimeError("VideoEdit streaming decode failed") from self._decode_error

    def materialize_window(self, input_indices: list[int]) -> tuple[list[Image.Image], list[Image.Image]]:
        if not input_indices:
            return [], []
        self._ensure_decoded_through(max(input_indices))
        frames = []
        masks = []
        for index in input_indices:
            entry = self._lookup_entry(index)
            frames.append(entry.resized_frame.copy())
            masks.append(entry.resized_mask.copy())
        return frames, masks

    def get_resized_frame(self, index: int) -> Image.Image:
        self._ensure_decoded_through(index)
        return self._lookup_entry(index).resized_frame.copy()

    def paste_back_frames(
        self,
        generated_frames: list[Image.Image],
        *,
        feather_px: int,
        adain_boundary_dilate: int,
    ) -> list[Image.Image]:
        resized_generated = [
            resize_frame(frame, self.crop_h, self.crop_w) for frame in generated_frames
        ]
        result: list[Image.Image] = []
        video_decoder = self._make_video_decoder()
        mask_decoder = self._make_mask_decoder()
        try:
            for global_index, generated_frame in enumerate(resized_generated):
                if global_index == 0 and self.reference_offset == 1:
                    assert self._reference_frame is not None
                    original_frame = self._reference_frame.copy()
                    raw_mask = Image.new("L", self.frame_size, 0)
                else:
                    original_frame = video_decoder.read_next()
                    raw_mask = mask_decoder.read_next()
                    if original_frame is None or raw_mask is None:
                        raise RuntimeError(
                            "Unexpected end of input while paste-back streaming "
                            f"at global index {global_index}"
                        )
                expanded_mask = expand_mask_frame(
                    raw_mask,
                    dilate_px=self.dilate_px,
                    scale=self.mask_scale,
                    force_zero=global_index == 0,
                )
                cropped_mask = crop_frame(expanded_mask, self.bbox)
                result.append(
                    paste_back_frame(
                        original_frame=original_frame,
                        generated_frame=generated_frame,
                        mask_frame=cropped_mask,
                        bbox=self.bbox,
                        feather_px=feather_px,
                        adain_boundary_dilate=adain_boundary_dilate,
                    )
                )
        finally:
            video_decoder.close()
            mask_decoder.close()
        return result

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5.0)
            self._prefetch_thread = None
        self._video_decoder.close()
        self._mask_decoder.close()
        self._cache.clear()
