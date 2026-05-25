from .video_info_ffmpeg import VideoInfoFfmpeg
import os
import ffmpeg
import json
import os
import ffmpeg
import subprocess
from .video_stream_writer import VideoStreamWriter
from pathlib import Path


# ffprobe -v quiet -select_streams v:0 -show_streams -print_format json test.mp4
class VideoStreamWriterFfmpeg(VideoStreamWriter):
    def __init__(
        self,
        save_path: str = None,
        writer: subprocess.Popen = None,
        refer_file: str = None,
        loglevel="warning",
        write_async: bool = True,
        enable: bool = True,
        **kwargs,
    ):
        self.write_async = write_async
        self.empty_stream = True
        self.is_close = True
        self.process = None  # type:subprocess.Popen
        self.path = save_path
        if not enable:
            return

        if self.path is not None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if writer is not None:
            self.process = writer
            self.is_close = False
            return

        args = {}
        for k, v in kwargs.items():
            args[k] = v

        profile = self.construct_profile(args, refer_file=refer_file)

        for k, v in self.get_default_arg().items():
            if k not in profile:
                profile[k] = v

        assert (
            "width" in profile and "height" in profile
        ), "need width * height while set writer of file"

        # print(f"profile: {profile}")

        addition_filter = {}
        if "cmd_scale" in profile:
            addition_filter["scale"] = profile["cmd_scale"]

        if "setfield" in profile:
            addition_filter["setfield"] = profile["setfield"]

        addtion_cmd = {}
        if "vcodec" in profile:
            if profile["vcodec"] == "prores_ks":
                assert (
                    "ffmpeg_profile_value" in profile
                ), "need ffmpeg_profile_value if vcodec==prores_ks"
                addtion_cmd["profile:v"] = profile["ffmpeg_profile_value"]

        cmd = ffmpeg.input(
            "pipe:",
            format=profile.get("format", "rawvideo"),
            pix_fmt=profile.get("rgb_fmt", "rgb24"),
            s="{}x{}".format(int(profile["width"]), int(profile["height"])),
            r=profile.get("fps", 30),
        )

        for filer in addition_filter:
            value = addition_filter[filer]
            if isinstance(value, dict):
                cmd = cmd.filter(filer, **value)
            else:
                cmd = cmd.filter(filer, value)

        if refer_file.endswith(".mxf") and profile["width"] == 3840 and profile["height"] == 2160 and profile["is_high_bit"]:
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!mxf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(profile.get("pix_fmt", "yuv422p10le"))
            self.process = cmd.output(
                save_path,
                vcodec=profile.get("vcodec", "libx264"),
                x264opts='avcintra-class=300:avcintra-flavor=sony:colorprim={}:transfer={}'.format(profile["color_primaries"], profile["color_transfer"]),
                pix_fmt=profile.get("pix_fmt", "yuv422p10le"),
                loglevel=loglevel,
                r=profile.get("fps", 30),
                **profile["cmd"],
                **profile["field_order_kwargs"],
                **{'avcintra-class': '300'},
            ).overwrite_output()  # type: subprocess.Popen
        else:
            self.process = cmd.output(
                save_path,
                vcodec=profile.get("vcodec", "libx264"),
                pix_fmt=profile.get("pix_fmt", "rgb24"),
                loglevel=loglevel,
                video_bitrate=profile.get("bit_rate", 10000000),
                r=profile.get("fps", 30),
                **addtion_cmd,
                **profile["cmd"],
                **profile["field_order_kwargs"],
            ).overwrite_output()  # type: subprocess.Popen

        self.process = self.process.run_async(pipe_stdin=True, cmd="/opt/conda/bin/ffmpeg")

        self.is_close = False

    def get_default_arg(self) -> dict:
        return {
            "format": "rawvideo",
            "pix_fmt": "yuv420p",
            "rgb_fmt": "rgb24",
            "fps": 30,
            "vcodec": "libx264",
            "bit_rate": 10000000,
            "field_order": "progressive",
            "field_order_kwargs": {},
        }

    def construct_profile(self, args: dict, refer_file=None) -> dict:
        profile = {}
        if refer_file is not None:
            profile = VideoInfoFfmpeg(refer_file).meta_data

        for k, v in args.items():
            profile[k] = v

        if "cmd_scale" not in profile:
            if profile.get("color_space", None) is not None:
                if "bt601" in profile["color_space"]:
                    color_matrix = "bt601"
                elif "bt470" in profile["color_space"]:
                    color_matrix = "bt470"
                elif 'bt2020' in profile["color_space"]:
                    color_matrix = 'bt2020'
                else:
                    color_matrix = profile["color_space"]
                cmd_scale = {
                    "in_color_matrix": "{}".format(color_matrix),
                    "out_color_matrix": "{}".format(color_matrix),
                }
                profile["cmd_scale"] = cmd_scale

        if "cmd" not in profile:
            cmd = {}
            if profile.get("color_space", None) is not None:
                cmd["colorspace"] = profile["color_space"]
            if profile.get("color_transfer", None) is not None:
                cmd["color_trc"] = profile["color_transfer"]
            if profile.get("color_primaries", None) is not None:
                cmd["color_primaries"] = profile["color_primaries"]
            # if profile.get("sample_aspect_ratio", None) is not None:
            #     cmd["sample_aspect_ratio"] = profile["sample_aspect_ratio"]
            # if profile.get("display_aspect_ratio", None) is not None:
            #     cmd["display_aspect_ratio"] = profile["display_aspect_ratio"]
            profile["cmd"] = cmd
        if "color_transfer" in profile["cmd"]:
            if profile["cmd"]["color_transfer"] == "reserved":
                profile["cmd"]["color_trc"] = "bt709"
        if "color_primaries" in profile["cmd"]:
            if profile["cmd"]["color_primaries"] == "reserved":
                profile["cmd"]["color_primaries"] = "bt709"
        if "color_trc" in profile["cmd"]:
            if profile["cmd"]["color_trc"] == "reserved":
                profile["cmd"]["color_trc"] = "bt709"

        if "color_transfer" in profile:
            if profile["color_transfer"] == "reserved":
                profile["color_trc"] = "bt709"
        if "color_primaries" in profile:
            if profile["color_primaries"] == "reserved":
                profile["color_primaries"] = "bt709"
        if "color_trc" in profile:
            if profile["color_trc"] == "reserved":
                profile["color_trc"] = "bt709"

        print("profile",profile)
        if refer_file is not None:
            probe = ffmpeg.probe(refer_file)
            video_streams = [
                stream for stream in probe["streams"] if stream["codec_type"] == "video"
            ]

            if "vcodec" not in profile:
                vcodec = video_streams[0]["codec_name"]
                if vcodec.lower() == "prores":
                    vcodec = "prores_ks"
                profile["vcodec"] = vcodec

            ffmpeg_profile_value_map = {
                "unknown": "3",
                "proxy": "0",
                "lt": "1",
                "standard": "2",
                "hq": "3",
                "4444": "4",
                "4444 xq": "5",
            }

            if "ffmpeg_profile_value" not in profile:
                profile["ffmpeg_profile_value"] = ffmpeg_profile_value_map.get(
                    video_streams[0].get("profile", "unknown").lower(), "3"
                )

            """
            Determine whether the video is progressive or interlaced scanning.
            return 'progressive', 'top_field_first', 'bottom_field_first' or 'unknown_profile["field_order"]'。
            If there is no video stream in the video or ffprobe fails, return 'error'.
            """
            try:
                if "field_order" not in profile:
                    field_order = video_streams[0].get("field_order")

                    if field_order in ["top", "top_field_first", "tb", "tt", "tff"]:
                        profile["field_order"] = "top_field_first"
                    elif field_order in [
                        "bottom",
                        "bottom_field_first",
                        "bb",
                        "bt",
                        "bff",
                    ]:
                        profile["field_order"] = "bottom_field_first"
                    elif field_order == "progressive":
                        profile["field_order"] = "progressive"
                    elif field_order == "interlaced":
                        profile["field_order"] = "interlaced"
                    elif field_order is None:
                        profile["field_order"] = "progressive"
                    # else:
                    #     return "unknown_profile["field_order"]"

            except ffmpeg.Error as e:
                print(f"FFprobe Error for {refer_file}: {e.stderr.decode('utf8')}")
                raise e
            except json.JSONDecodeError:
                print(f"Failed to decode JSON from ffprobe output for {refer_file}")
                raise e
            except Exception as e:
                print(f"An unexpected error occurred during scan type detection: {e}")
                raise e

            if "field_order_kwargs" not in profile and "field_order" in profile:
                profile["field_order_kwargs"] = {}
                if profile["field_order"] in [
                    "top_field_first",
                    "bottom_field_first",
                    "interlaced",
                ]:
                    profile["field_order_kwargs"]["flags"] = "+ildct+ilme"
                    if profile["field_order"] == "top_field_first":
                        profile["field_order_kwargs"]["top"] = 1
                        # profile["field_order_kwargs"]["vf"] = "setfield=tff"
                    elif profile["field_order"] == "bottom_field_first":
                        profile["field_order_kwargs"]["top"] = 0
                        # profile["field_order_kwargs"]["vf"] = "setfield=bff"
                elif profile["field_order"] == "progressive":
                    pass
                else:
                    print(
                        f"Warning: Unrecognized scan type {profile['field_order']}. Defaulting to progressive processing."
                    )
                profile = self.field_order_compatible(profile)

        return profile

    def field_order_compatible(self, profile: dict) -> dict:
        if VideoStreamWriterFfmpeg.get_ffmpeg_version() >= 7:
            if "top" in profile.get("field_order_kwargs", {}):
                if profile["field_order_kwargs"]["top"] == 1:
                    del profile["field_order_kwargs"]["top"]
                    profile["setfield"] = "tff"
                elif profile["field_order_kwargs"]["top"] == 0:
                    del profile["field_order_kwargs"]["top"]
                    profile["setfield"] = "bff"
                else:
                    raise ValueError(
                        f'unknown -top={profile["field_order_kwargs"]["top"]} for early ffmpeg version to -vf'
                    )
        return profile

    def get_ffmpeg_version() -> int:
        import subprocess
        import re

        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], stdout=subprocess.PIPE, text=True
            )
            match = re.search(r"ffmpeg version (\d+)\.", result.stdout)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return -1

    @staticmethod
    def create_writer(
        save_path: str,
        width: int,
        height: int,
        fps: 25,
        vcodec: str = "libx264",
        encode_fmt: str = "yuv420p",
        input_fmt: str = "rgb24",
        crf: int = 28,
        loglevel="warning",
        resize_to: tuple = None,
    ) -> subprocess.Popen:
        os.makedirs(Path(save_path).parent.resolve(), exist_ok=True)
        tmp = ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt=input_fmt,
            s=f"{width}x{height}",
            r=fps,
        )

        if resize_to is not None:
            tmp = tmp.filter("scale", resize_to[0], resize_to[1])

        process = (
            tmp.output(
                f"{save_path}",
                pix_fmt=encode_fmt,
                vcodec=vcodec,
                r=fps,
                loglevel=loglevel,
                crf=crf,
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        return process

    @staticmethod
    def add_audio_to_video(video_path, audio_path, output_path):
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-filter_complex",
            "[1:a]apad",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-loglevel",
            "quiet",
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    def set_process(self, writer: subprocess.Popen):
        self.process = writer

    def Write(self, data: bytes, return_info: bool = False) -> bool | tuple[bool, str]:
        if self.is_close:
            if not return_info:
                return False
            else:
                return False, "Write to a closed stream"

        try:
            self.process.stdin.write(data)
            if not self.write_async:
                self.process.stdin.flush()
            self.empty_stream = False
        except Exception as ex:
            print(f"error: ffmpeg write meet error, ex info: {ex}")
            if not return_info:
                return False
            else:
                return False, f"error: ffmpeg write meet error, ex info: {ex}"
        if not return_info:
            return True
        else:
            return True, ""

    def Close(self):
        if not self.is_close:
            self.process.stdin.close()

            if not self.empty_stream:
                self.process.wait()
            self.is_close = True

    def __del__(self):
        self.Close()

