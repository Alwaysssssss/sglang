# SPDX-License-Identifier: Apache-2.0
"""Launch the native SGLang HTTP server with optimized VSR model residency."""

import argparse
import os
import signal

from sglang.multimodal_gen.configs.pipeline_configs.vsr import WanVSRPipelineConfig
from sglang.multimodal_gen.runtime.launch_server import kill_process_tree, launch_server
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--wan-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30176)
    parser.add_argument("--output-dir", default="output_results/vsr/server_api")
    parser.add_argument("--tile-devices", nargs="+", default=None)
    parser.add_argument("--vae-cpu-offload", action="store_true")
    offload = parser.add_mutually_exclusive_group()
    offload.add_argument("--dit-cpu-offload", action="store_true")
    offload.add_argument("--dit-layerwise-offload", action="store_true")
    parser.add_argument("--dit-offload-prefetch-size", type=float, default=0.0)
    args = parser.parse_args()
    config = WanVSRPipelineConfig(
        precision="bfloat16",
        tile_devices=args.tile_devices,
        cudnn_benchmark=True,
        channels_last_3d=True,
        compile_encoder=True,
        compile_decoder=True,
        decoder_implicit_padding=True,
        gpu_postprocess=True,
    )
    server = ServerArgs.from_kwargs(
        model_path=args.checkpoint_dir,
        pipeline_class_name="WanVSRPipeline",
        pipeline_config=config,
        component_paths={"wan_root": args.wan_root},
        host=args.host,
        port=args.port,
        num_gpus=1,
        trust_remote_code=True,
        output_path=args.output_dir,
        dit_cpu_offload=args.dit_cpu_offload,
        dit_layerwise_offload=args.dit_layerwise_offload,
        vae_cpu_offload=args.vae_cpu_offload,
        dit_offload_prefetch_size=args.dit_offload_prefetch_size,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
    )

    def stop(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    try:
        launch_server(server)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
