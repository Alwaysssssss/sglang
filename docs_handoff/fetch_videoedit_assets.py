#!/usr/bin/env python3
"""拉取一次 videoedit 请求用到的四份素材（输入视频 / mask / 参考图 / 输出视频）。

素材位于腾讯云 COS（`vrs-mms-1258229344`，prefix `/flowcut`），桶不公开读，
所以这里用 AWS SigV4 对每个请求签名。仅依赖标准库，无需 boto3：

    python3 docs_handoff/fetch_videoedit_assets.py                # 拉到 docs_handoff/assets/
    python3 docs_handoff/fetch_videoedit_assets.py --out-dir /tmp/run1 --force
    python3 docs_handoff/fetch_videoedit_assets.py --assets output_video
    python3 docs_handoff/fetch_videoedit_assets.py --video-url flowcut/test/.../other.mov

默认素材来自 docs_handoff/api_server.md 里的那次请求；桶/AK/SK 均可用参数或环境变量
覆盖（COS_ACCESS_KEY / COS_SECRET_KEY / COS_BUCKET / COS_ENDPOINT / COS_REGION）。
AK/SK 必须通过参数或环境变量提供，代码中不保存凭证。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# --- 默认素材（api_server.md 中那次请求的输入与产物） ---------------------------------
DEFAULT_SOURCES = {
    "input_video": "flowcut/test/2026/07/16/b5d7c82008aecc1ee5f1d3ac816473b1d9cb1741.mov",
    "mask": "flowcut/test/2026/07/16/d7395086afc2116bb87b0c11400836142d903d81.json",
    "reference_image": "flowcut/test/2026/07/16/fcbd87e6624ece681c1d863d9a89957a8ef717de.png",
    "output_video": "flowcut/2026/09/25/075417_videoedit-bb50e3b46e934d5bb507de6e839bf2d1.mov",
}
FILENAMES = {
    "input_video": "input_video.mov",
    "mask": "mask.json",
    "reference_image": "reference_image.png",
    "output_video": "output_video.mov",
}
DESCRIPTIONS = {
    "input_video": "请求入参 videoUrl",
    "mask": "请求入参 maskUrl（flowcut RLE 帧 mask JSON）",
    "reference_image": "请求入参 referenceImageUrl",
    "output_video": "接口返回的输出视频 URL",
}

DEFAULT_ENDPOINT = "cos.ap-beijing.myqcloud.com"
DEFAULT_BUCKET = "vrs-mms-1258229344"
DEFAULT_REGION = "ap-beijing"
DEFAULT_PREFIX = "/flowcut"


_CHUNK = 1 << 20


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --- SigV4 签名（COS 兼容 AWS SigV4，必须用 x-amz-date/x-amz-content-sha256） --------
class CosClient:
    def __init__(self, endpoint: str, bucket: str, region: str, access_key: str, secret_key: str):
        self.endpoint = endpoint.strip().rstrip("/")
        if not self.endpoint.startswith(("http://", "https://")):
            self.endpoint = "https://" + self.endpoint
        self.host = urllib.parse.urlparse(self.endpoint).netloc
        self.bucket = bucket
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key

    def _sign(self, method: str, object_key: str) -> urllib.request.Request:
        now = datetime.datetime.now(datetime.timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")

        canonical_uri = "/" + urllib.parse.quote(f"{self.bucket}/{object_key}", safe="/~")
        payload_hash = hashlib.sha256(b"").hexdigest()
        canonical_headers = (
            f"host:{self.host}\n"
            f"x-amz-content-sha256:{payload_hash}\n"
            f"x-amz-date:{amz_date}\n"
        )
        signed_headers = "host;x-amz-content-sha256;x-amz-date"
        canonical_request = (
            f"{method}\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        )

        scope = f"{date_stamp}/{self.region}/s3/aws4_request"
        string_to_sign = (
            f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n"
            f"{hashlib.sha256(canonical_request.encode()).hexdigest()}"
        )
        key = ("AWS4" + self.secret_key).encode()
        for part in (date_stamp, self.region, "s3", "aws4_request"):
            key = hmac.new(key, part.encode(), hashlib.sha256).digest()
        signature = hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()

        return urllib.request.Request(
            f"{self.endpoint}{canonical_uri}",
            method=method,
            headers={
                "x-amz-content-sha256": payload_hash,
                "x-amz-date": amz_date,
                "Authorization": (
                    f"AWS4-HMAC-SHA256 Credential={self.access_key}/{scope}, "
                    f"SignedHeaders={signed_headers}, Signature={signature}"
                ),
            },
        )

    def open(self, method: str, object_key: str, timeout: float = 60.0):
        request = self._sign(method, object_key)
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # HEAD 的响应没有 body，取不到 COS 的 XML 错误码时就给个通用提示
            detail = _cos_error(exc) or "对象不存在 / 无权访问 / 签名被拒"
            raise RuntimeError(f"{method} {object_key} 失败: HTTP {exc.code} {detail}") from exc

    def head_size(self, object_key: str) -> tuple[int, str]:
        with self.open("HEAD", object_key) as response:
            return int(response.headers.get("Content-Length") or 0), response.headers.get(
                "Content-Type"
            ) or "?"

    def download(self, object_key: str, target: str, show_progress: bool = True) -> tuple[int, str]:
        """流式下载，返回 (字节数, sha256)。先写 .part 再改名，避免半截文件。"""
        tmp = target + ".part"
        digest = hashlib.sha256()
        total = 0
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with self.open("GET", object_key) as response, open(tmp, "wb") as handle:
            expected = int(response.headers.get("Content-Length") or 0)
            last_tick = 0.0
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                now = time.monotonic()
                if show_progress and expected and now - last_tick > 0.3:
                    last_tick = now
                    _progress(total, expected)
        if show_progress:
            sys.stderr.write("\r" + " " * 60 + "\r")
        os.replace(tmp, target)
        return total, digest.hexdigest()


def _progress(done: int, total: int) -> None:
    pct = done * 100.0 / total if total else 0.0
    sys.stderr.write(f"\r    下载中 {pct:5.1f}%  {_human(done)} / {_human(total)}")
    sys.stderr.flush()


def _human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} GB"


def _cos_error(exc: urllib.error.HTTPError) -> str:
    """把 COS 的 XML 错误体压成一行，便于排查权限/签名问题。"""
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - 读不到就算了
        return ""
    code = message = ""
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("<Code>"):
            code = line[6:-7]
        elif line.startswith("<Message>"):
            message = line[9:-10]
    return f"[{code}] {message}".strip()


def resolve_object_key(source: str, bucket: str, prefix: str) -> str:
    """支持裸对象键 / https URL / s3:// URL，并按服务端规则补 prefix（见 storage.py）。"""
    source = source.strip()
    if source.startswith("s3://"):
        parsed = urllib.parse.urlparse(source)
        key = urllib.parse.unquote(parsed.path)
    elif source.lower().startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(source)
        key = urllib.parse.unquote(parsed.path)
        parts = [part for part in key.split("/") if part]
        if parts and parts[0] == bucket:  # URL 里带 bucket 段时去掉，避免重复
            parts = parts[1:]
        key = "/".join(parts)
    else:
        key = source

    key = key.strip().lstrip("/")
    if not key or ".." in key.split("/"):
        raise ValueError(f"非法对象键: {source!r}")

    normalized_prefix = (prefix or "").strip().strip("/")
    if normalized_prefix and key != normalized_prefix and not key.startswith(f"{normalized_prefix}/"):
        key = f"{normalized_prefix}/{key}"
    return key


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="从 COS 拉取一次 videoedit 请求的输入视频 / mask / 参考图 / 输出视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", default=os.path.join(repo_root, "docs_handoff", "assets"),
                        help="下载目录（默认 docs_handoff/assets）")
    parser.add_argument("--assets", default="all",
                        help="逗号分隔的素材名，可选 all / " + " / ".join(DEFAULT_SOURCES))
    parser.add_argument("--force", action="store_true", help="本地已存在且大小一致时也重新下载")
    parser.add_argument("--no-manifest", action="store_true", help="不写 manifest.json")
    parser.add_argument("--endpoint", default=os.getenv("COS_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--bucket", default=os.getenv("COS_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--region", default=os.getenv("COS_REGION", DEFAULT_REGION))
    parser.add_argument("--prefix", default=os.getenv("COS_PREFIX", DEFAULT_PREFIX),
                        help="对象键前缀，默认 /flowcut；传空字符串表示不加前缀")
    parser.add_argument("--access-key", default=os.getenv("COS_ACCESS_KEY"))
    parser.add_argument("--secret-key", default=os.getenv("COS_SECRET_KEY"))
    parser.add_argument("--video-url", default=None, help="覆盖输入视频来源（裸键或 URL）")
    parser.add_argument("--mask-url", default=None, help="覆盖 mask 来源")
    parser.add_argument("--reference-url", default=None, help="覆盖参考图来源")
    parser.add_argument("--output-url", default=None, help="覆盖输出视频来源")
    args = parser.parse_args()

    if not args.access_key or not args.secret_key:
        parser.error("请设置 COS_ACCESS_KEY 和 COS_SECRET_KEY 环境变量，或提供 --access-key 和 --secret-key")

    sources = dict(DEFAULT_SOURCES)
    for name, override in (
        ("input_video", args.video_url),
        ("mask", args.mask_url),
        ("reference_image", args.reference_url),
        ("output_video", args.output_url),
    ):
        if override:
            sources[name] = override

    if args.assets.strip() == "all":
        selected = list(sources)
    else:
        selected = [name.strip() for name in args.assets.split(",") if name.strip()]
        unknown = [name for name in selected if name not in sources]
        if unknown:
            parser.error(f"未知素材名: {', '.join(unknown)}（可选 {', '.join(sources)}）")

    client = CosClient(args.endpoint, args.bucket, args.region, args.access_key, args.secret_key)
    os.makedirs(args.out_dir, exist_ok=True)

    manifest: list[dict] = []
    failed = 0
    log(f"输出目录: {args.out_dir}")
    log(f"存储桶:   {client.endpoint}/{args.bucket}  (prefix={args.prefix or '<无>'}, region={args.region})")
    log("")

    for name in selected:
        target = os.path.join(args.out_dir, FILENAMES[name])
        log(f"[{name}] {DESCRIPTIONS[name]}")
        try:
            object_key = resolve_object_key(sources[name], args.bucket, args.prefix)
            log(f"    对象键 {object_key}")
            remote_size, content_type = client.head_size(object_key)
            local_size = os.path.getsize(target) if os.path.exists(target) else -1
            if local_size == remote_size and not args.force:
                log(f"    已存在且大小一致（{_human(remote_size)}），跳过；--force 可重下")
                sha = _sha256_file(target)
            else:
                if local_size >= 0:
                    log(f"    重新下载（本地 {_human(max(local_size, 0))}，远端 {_human(remote_size)}）")
                size, sha = client.download(object_key, target)
                if remote_size and size != remote_size:
                    raise RuntimeError(f"大小不符：收到 {size}，期望 {remote_size}")
                remote_size = size
                log(f"    完成 → {target}  ({_human(size)})")
            manifest.append({
                "name": name,
                "object_key": object_key,
                "url": f"{client.endpoint}/{args.bucket}/{object_key}",
                "local_path": target,
                "size": remote_size,
                "sha256": sha,
                "content_type": content_type,
                "description": DESCRIPTIONS[name],
            })
        except Exception as exc:  # noqa: BLE001 - 逐个素材报错，不中断其余素材
            failed += 1
            log(f"    !! {exc}")
        log("")

    if not args.no_manifest and manifest:
        path = os.path.join(args.out_dir, "manifest.json")
        # 与已有清单合并：只拉部分素材时不要把其余记录冲掉
        previous = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    previous = {entry["name"]: entry for entry in json.load(handle).get("assets", [])}
            except (OSError, ValueError, KeyError):
                log("    已有 manifest.json 无法解析，将整体重写")
        merged = [*previous.values()]
        for entry in manifest:
            merged = [item for item in merged if item.get("name") != entry["name"]] + [entry]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "endpoint": client.endpoint, "bucket": args.bucket,
                       "assets": merged}, handle, ensure_ascii=False, indent=2)
        log(f"清单已写入 {path}（{len(merged)} 项）")

    log("")
    log(f"{'素材':<16}{'大小':>12}  {'sha256':<16} 本地文件")
    for entry in manifest:
        log(f"{entry['name']:<16}{_human(entry['size']):>12}  {entry['sha256'][:16]} {entry['local_path']}")
    if failed:
        log(f"\n{failed} 个素材失败")
    return 1 if failed else 0


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
